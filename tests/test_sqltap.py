# -*- encoding: utf8 -*-
from __future__ import print_function

import collections
import os
import tempfile
import uuid
import warnings
from dataclasses import dataclass

import pytest
import sqlalchemy.event
import sqlparse
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import text
from werkzeug.test import Client
from werkzeug.testapp import test_app
from werkzeug.wrappers import Response

import sqltap
import sqltap.wsgi

warnings.simplefilter(os.environ.get("WARNING_ACTION", "error"))

REPORT_TITLE = "SQLTap Profiling Report"


def _startswith(qs, text):
    return list(filter(lambda q: str(q.text).strip().startswith(text), qs))


# class MockResults(object):
#     def __init__(self, rowcount):
#         rowcount = rowcount


@dataclass
class MockResults:
    rowcount: int


Base = declarative_base()


class A(Base):
    __tablename__ = "a"
    id = Column("id", Integer, primary_key=True)
    name = Column("name", String)
    description = Column("description", String)


class B(Base):
    __tablename__ = "b"
    id = Column("id", Integer, primary_key=True)


@pytest.fixture
def setup_engine() -> tuple:
    engine = create_engine("sqlite:///:memory:", echo=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)
    return session, engine


@pytest.fixture
def setup_engine2():
    engine = create_engine("sqlite:///:memory:", echo=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)
    return session, engine


def check_report(report):
    """
    Check whether the SQL report was actually rendered.
    If the report fails, then a Mako error report is generated instead
    with `Mako Runtime Error` in its title.
    """
    assert REPORT_TITLE in report


def assertEqual(expected, actual, message=None):
    message = message or "{0!r} == {1!r}".format(expected, actual)
    assert expected == actual, message


def test_insert(setup_engine):
    """Simple test that sqltap collects an insert query."""
    session, engine = setup_engine
    profiler = sqltap.start(engine)

    sess = session()
    sess.add(A())
    sess.flush()

    stats = profiler.collect()
    assert len(_startswith(stats, "INSERT")) == 1
    profiler.stop()


def test_select(setup_engine):
    """Simple test that sqltap collects a select query."""
    session, engine = setup_engine
    profiler = sqltap.start(engine)

    sess = session()
    sess.query(A).all()

    stats = profiler.collect()
    assert len(_startswith(stats, "SELECT")) == 1
    profiler.stop()


def test_engine_scoped(setup_engine, setup_engine2):
    """
    Test that calling sqltap.start with a particular engine instance
    properly captures queries only to that engine.
    """
    session, _ = setup_engine
    session2, engine2 = setup_engine2
    profiler = sqltap.start(engine2)

    sess = session()
    sess.query(A).all()

    sess2 = session2()
    sess2.query(B).all()

    stats = _startswith(profiler.collect(), "SELECT")
    assert len(stats) == 1
    profiler.stop()


def test_engine_global(setup_engine, setup_engine2):
    """
    Test that registering globally for all queries correctly pulls queries
    from multiple engines.
    """
    session, _ = setup_engine
    session2, _ = setup_engine2
    profiler = sqltap.start()

    sess = session()
    sess.query(A).all()

    sess2 = session2()
    sess2.query(B).all()

    stats = _startswith(profiler.collect(), "SELECT")
    assert len(stats) == 2
    profiler.stop()


def test_start_twice(setup_engine):
    """
    Ensure that multiple calls to ProfilingSession.start() raises assertion
    error.
    """
    _, engine = setup_engine
    profiler = sqltap.ProfilingSession(engine)
    profiler.start()
    try:
        profiler.start()
        raise ValueError("Second start should have asserted")
    except AssertionError:
        pass
    profiler.stop()


def test_stop(setup_engine):
    """
    Ensure queries after you call ProfilingSession.stop() are not recorded.
    """
    session, engine = setup_engine
    profiler = sqltap.start(engine)
    sess = session()
    sess.query(A).all()
    profiler.stop()
    sess.query(A).all()

    assert len(profiler.collect()) == 1


def test_stop_global(setup_engine):
    """
    Ensure queries after you call ProfilingSession.stop() are not recorded
    when passing in the 'global' Engine object to record queries across all
    engines.
    """
    session, _ = setup_engine
    profiler = sqltap.start()
    sess = session()
    sess.query(B).all()
    profiler.stop()
    sess.query(B).all()

    assert len(profiler.collect()) == 1


def test_querygroup_add_params_no_dup():
    """Ensure that two identical parameter sets, belonging to different queries,
    are treated as separate."""
    python_query = "SELECT * FROM pythons WHERE name=:name"
    directors_query = "SELECT * FROM movies WHERE director=:name"
    jones = {"name": "Terry Jones"}
    gilliam = {"name": "Terry Gilliam"}
    query_groups = collections.defaultdict(sqltap.QueryGroup)
    all_group = sqltap.QueryGroup()

    def add(query, params, stack, rowcount, start=1, end=2):
        stats = sqltap.QueryStats(
            query, stack, start, end, None, params, MockResults(rowcount)
        )
        query_groups[stack].add(stats)
        all_group.add(stats)
        return stats

    add(python_query, jones, "stack1", 1)
    add(directors_query, jones, "stack2", 4)
    add(python_query, gilliam, "stack1", 1)
    add(directors_query, gilliam, "stack2", 12)
    add(python_query, gilliam, "stack1", 1)
    add(directors_query, gilliam, "stack9", 12)

    assertEqual(1 + 1 + 1, query_groups["stack1"].rowcounts)
    assertEqual(4 + 12, query_groups["stack2"].rowcounts)
    assertEqual(12, query_groups["stack9"].rowcounts)
    assertEqual((1 + 1 + 1) + (4 + 12) + 12, all_group.rowcounts)

    assertEqual(3, len(all_group.stacks))
    assertEqual(set([1, 2, 3]), set(all_group.stacks.values()))

    assertEqual(4, len(all_group.params_hashes))
    gilliam_movie_queries = all_group.params_hashes[
        (hash(directors_query), sqltap.QueryStats.calculate_params_hash(gilliam))
    ]
    jones_movie_queries = all_group.params_hashes[
        (hash(directors_query), sqltap.QueryStats.calculate_params_hash(jones))
    ]
    assertEqual(1, jones_movie_queries[0])
    assertEqual(jones, jones_movie_queries[2])
    assertEqual(2, gilliam_movie_queries[0])
    assertEqual(gilliam, gilliam_movie_queries[2])


def test_query_stats_with_no_hashable_params():
    """Regression test for when sql query params contain un-hashable python
    object e.g. Postgres ARRAY -> list.
    """
    params = {"tags": ["programming", "python", "sqla"]}

    assertEqual(
        sqltap.QueryStats.calculate_params_hash(params),
        sqltap.QueryStats.calculate_params_hash(params),
    )


def test_report(setup_engine):
    session, engine = setup_engine
    profiler = sqltap.start(engine)

    sess = session()
    q = sess.query(A)
    qtext = sqltap.format_sql(str(q))
    q.all()

    stats = profiler.collect()
    report = sqltap.report(stats, report_format="html")
    assert REPORT_TITLE in report
    assert qtext in report
    report = sqltap.report(stats, report_format="text")
    assert REPORT_TITLE in report
    assert sqlparse.format(qtext, reindent=True) in report
    profiler.stop()


def test_report_to_file(setup_engine):
    session, engine = setup_engine
    profiler = sqltap.start(engine)

    sess = session()
    # q = sess.query(A).filter(A.name == "معاذ")
    q = sess.query(A).filter(A.name == "Odin")
    q.all()
    stats = profiler.collect()

    fd, temp_path = tempfile.mkstemp()
    os.close(fd)
    report = sqltap.report(stats, filename=temp_path)

    with open(temp_path) as fp:
        assertEqual(report, fp.read())


def test_report_raw_sql(setup_engine):
    """Ensure that reporting works when raw SQL queries were emitted."""
    session, engine = setup_engine
    profiler = sqltap.start(engine)

    sess = session()
    sql = f"SELECT * FROM {A.__tablename__}"
    with engine.connect() as conn:
        conn.execute(text(sql))

    stats = profiler.collect()
    report = sqltap.report(stats, report_format="html")
    assert REPORT_TITLE in report
    assert sqltap.format_sql(sql) in report
    report = sqltap.report(stats, report_format="text")
    # assert REPORT_TITLE in report
    # assert sqlparse.format(sql, reindent=True) in report
    # profiler.stop()


def test_report_ddl(setup_engine2):
    """Ensure that reporting works when DDL were emitted"""
    _, engine = setup_engine2
    profiler = sqltap.start(engine)

    stats = profiler.collect()
    report = sqltap.report(stats, report_format="html")
    assert REPORT_TITLE in report
    report = sqltap.report(stats, report_format="text")
    assert REPORT_TITLE in report
    profiler.stop()


def test_no_before_exec(setup_engine):
    """
    If SQLTap is started dynamically on one thread,
    any SQLAlchemy sessions running on other threads start being profiled.
    Their connections did not receive the before_execute event,
    so when they receive the after_execute event, extra care must be taken.
    """
    session, engine = setup_engine
    profiler = sqltap.ProfilingSession(engine)
    sqlalchemy.event.listen(engine, "after_execute", profiler._after_exec)
    sess = session()
    q = sess.query(A)
    q.all()
    stats = profiler.collect()
    assert len(stats) == 1
    assert stats[0].duration == 0.0, str(stats[0].duration)
    sqlalchemy.event.remove(engine, "after_execute", profiler._after_exec)


def test_report_aggregation(setup_engine):
    """
    Test that we aggregate stats for the same query called from
    different locations as well as aggregating queries called
    from the same stack trace.
    """
    session, engine = setup_engine
    profiler = sqltap.start(engine)

    sess = session()
    q = sess.query(A)
    q.all()
    q.all()
    q2 = sess.query(A).filter(A.id == 10)
    for _ in range(10):
        q2.all()

    report = sqltap.report(profiler.collect())
    print(report)
    assert "2 unique" in report
    assert "<dd>10</dd>" in report
    profiler.stop()


def test_report_aggregation_w_different_param_sets(setup_engine):
    """
    Test that report rendering works with groups of queries
    containing different parameter sets
    """
    session, engine = setup_engine
    sess = session()

    a1 = A(name=uuid.uuid4().hex, description="")
    a2 = A(name=uuid.uuid4().hex, description="")
    sess.add_all([a1, a2])
    sess.commit()

    a1 = sess.get(A, a1.id)
    a2 = sess.get(A, a2.id)

    profiler = sqltap.start(engine)
    # this will create queries with the same text, but different param sets
    # (different query.params.keys() collections)
    a1.name = uuid.uuid4().hex
    a2.description = uuid.uuid4().hex
    sess.flush()

    report = sqltap.report(profiler.collect())
    print(report)
    profiler.stop()
    check_report(report)


def test_start_stop(setup_engine):
    session, engine = setup_engine
    sess = session()
    q = sess.query(A)
    profiled = sqltap.ProfilingSession(engine)

    q.all()

    stats = profiled.collect()
    assert len(_startswith(stats, "SELECT")) == 0

    profiled.start()
    q.all()
    q.all()
    profiled.stop()
    q.all()

    stats2 = profiled.collect()
    assert len(_startswith(stats2, "SELECT")) == 2


def test_decorator(setup_engine):
    """Test that queries issued in a decorated function are profiled"""
    session, engine = setup_engine
    sess = session()
    q = sess.query(A)
    profiled = sqltap.ProfilingSession(engine)

    @profiled
    def test_function():
        session().query(A).all()

    q.all()

    stats = profiled.collect()
    assert len(_startswith(stats, "SELECT")) == 0

    test_function()
    test_function()
    q.all()

    stats = profiled.collect()
    assert len(_startswith(stats, "SELECT")) == 2


def test_context_manager(setup_engine):
    session, engine = setup_engine
    sess = session()
    q = sess.query(A)
    profiled = sqltap.ProfilingSession(engine)

    with profiled:
        q.all()

    q.all()

    stats = profiled.collect()
    assert len(_startswith(stats, "SELECT")) == 1


def test_context_fn(setup_engine):
    session, engine = setup_engine
    profiler = sqltap.start(engine, lambda *args: 1)

    sess = session()
    q = sess.query(A)
    q.all()
    stats = profiler.collect()

    ctxs = [qstats.user_context for qstats in _startswith(stats, "SELECT")]
    assert ctxs[0] == 1
    profiler.stop()


def test_context_fn_isolation(setup_engine):
    session, engine = setup_engine
    x = {"i": 0}

    def context_fn(*args):
        x["i"] += 1
        return x["i"]

    profiler = sqltap.start(engine, context_fn)

    sess = session()
    sess.query(A).all()

    sess2 = session()
    sess2.query(A).all()

    stats = profiler.collect()

    ctxs = [qstats.user_context for qstats in _startswith(stats, "SELECT")]
    assert ctxs.count(1) == 1
    assert ctxs.count(2) == 1
    profiler.stop()


def test_collect_empty(setup_engine):
    session, engine = setup_engine
    profiler = sqltap.start(engine)
    assert len(profiler.collect()) == 0
    profiler.stop()


def test_collect_fn(setup_engine):
    session, engine = setup_engine
    collection = []

    def my_collector(q):
        collection.append(q)

    sess = session()
    profiler = sqltap.start(engine, collect_fn=my_collector)
    sess.query(A).all()

    assert len(collection) == 1

    sess.query(A).all()

    assert len(collection) == 2
    profiler.stop()


def test_collect_fn_execption_on_collect(setup_engine):
    _, engine = setup_engine

    def noop():
        pass

    profiler = sqltap.start(engine, collect_fn=noop)
    with pytest.raises(AssertionError):
        profiler.collect()
    profiler.stop()


def test_report_escaped(setup_engine2):
    """Test that string escaped correctly."""
    session, engine2 = setup_engine2
    profiler = sqltap.start(engine2)
    sess = session()
    sess.query(B).filter(B.id == "<blockquote class='test'>").all()

    report = sqltap.report(profiler.collect())
    assert "<blockquote class='test'>" not in report
    assert "&#34;&lt;blockquote class=&#39;test&#39;&gt;&#34;" in report
    profiler.stop()


def test_context_return_self():
    with sqltap.ProfilingSession() as profiler:
        assert type(profiler) is sqltap.ProfilingSession


@pytest.fixture
def setup_client() -> tuple:
    app = sqltap.wsgi.SQLTapMiddleware(app=test_app)
    client = Client(app, Response)
    return app, client


def test_can_construct_wsgi_wrapper(setup_client):
    """
    Only verifies that the imports and __init__ work, not a real Test.
    """
    app, _ = setup_client
    sqltap.wsgi.SQLTapMiddleware(app)


def test_wsgi_get_request(setup_client):
    """Verify we can get the middleware path"""
    app, client = setup_client
    response = client.get(app.path)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_wsgi_post_turn_on(setup_client):
    """Verify we can POST turn=on to middleware"""
    app, client = setup_client
    response = client.post(app.path, data="turn=on")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_wsgi_post_turn_off(setup_client):
    """Verify we can POST turn=off to middleware"""
    app, client = setup_client
    response = client.post(app.path, data="turn=off")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_wsgi_post_turn_400(setup_client):
    """Verify we POSTing and invalid turn value returns a 400"""
    app, client = setup_client
    response = client.post(app.path, data="turn=invalid_string")
    assert response.status_code == 400
    assert "text/plain" in response.headers["content-type"]


def test_wsgi_post_clear(setup_client):
    """Verify we can POST clean=1 works"""
    app, client = setup_client
    response = client.post(app.path, data="clear=1")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
