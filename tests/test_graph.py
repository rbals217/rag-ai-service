import os

os.environ["SERVICE_TOKEN"] = "test-token"
os.environ["RAG_ENGINE"] = "stub"

import pytest
from fastapi.testclient import TestClient

from app import app
from engine import ABSTAIN_MESSAGE, StubEngine
from graph import build_graph, run_graph
from settings import Settings


HEADERS = {"X-Service-Token": os.environ["SERVICE_TOKEN"]}


# 테스트용 공연 데이터 생성
@pytest.fixture
def graph_and_engine():
    settings = Settings()
    engine = StubEngine(settings)

    # 공연 1번
    engine.index_concert(
        1,
        "뮤지컬 파이썬",
        50000,
        "서울",
        "공연 공지사항",
        "홍길동",
        "파이썬 공연은 매주 목요일 저녁 7시에 시작합니다.",
        "일반석 50000원",
        "공연 시작 30분 전 입장",
        "취소는 공연 3일 전까지 가능합니다.",
        "프로그래밍을 소재로 한 뮤지컬",
        "밝고 유쾌한 분위기"
    )

    # 공연 2번
    engine.index_concert(
        2,
        "클래식 음악회",
        70000,
        "부산",
        "음악회 공지사항",
        "김철수",
        "클래식 음악회는 매주 금요일 저녁 8시에 시작합니다.",
        "일반석 70000원",
        "공연 시작 20분 전 입장",
        "취소는 공연 5일 전까지 가능합니다.",
        "아름다운 클래식 음악 공연",
        "차분하고 웅장한 분위기"
    )

    return build_graph(engine, settings), engine


# 정상적으로 검색하고 답변하는지 확인
def test_generate_path(graph_and_engine):
    graph, engine = graph_and_engine

    r = run_graph(
        graph,
        "파이썬 공연은 언제 시작해?",
        [1, 2],
        4
    )

    assert r["abstained"] is False
    assert r["trace"] == [
        "retrieve",
        "grade",
        "generate",
        "format"
    ]
    assert r["sources"][0]["concert_id"] == 1


# 검색 결과가 없으면 질문 재작성 후 거절하는지 확인
def test_rewrite_then_abstain(graph_and_engine):
    graph, engine = graph_and_engine

    r = run_graph(
        graph,
        "블루베리 우주선 조립법",
        [1, 2],
        4
    )

    assert r["abstained"] is True
    assert r["answer"] == ABSTAIN_MESSAGE
    assert "rewrite" in r["trace"]
    assert r["trace"][-2:] == ["abstain", "format"]
    assert r["sources"] == []


# 공연 ID 목록이 비어 있으면 즉시 거절하는지 확인
def test_empty_allowlist_abstains_immediately(graph_and_engine):
    graph, engine = graph_and_engine

    r = run_graph(
        graph,
        "파이썬 공연은 언제 시작해?",
        [],
        4
    )

    assert r["abstained"] is True
    assert r["trace"] == [
        "retrieve",
        "grade",
        "abstain",
        "format"
    ]


# 허용된 공연 ID만 검색하는지 확인
def test_allowlist_is_respected(graph_and_engine):
    graph, engine = graph_and_engine

    r = run_graph(
        graph,
        "파이썬 공연은 언제 시작해?",
        [2],
        4
    )

    assert all(
        source["concert_id"] != 1
        for source in r["sources"]
    )


# FastAPI 테스트 클라이언트
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# 서비스 토큰이 없으면 접근을 차단하는지 확인
def test_token_required(client):
    r = client.post(
        "/query",
        json={
            "question": "공연 정보 알려줘",
            "concertIds": [1]
        }
    )

    assert r.status_code == 401


# 공연 색인 및 검색 API 확인
def test_index_and_query_api(client):

    # 공연 색인
    r = client.post(
        "/index/concert",
        headers=HEADERS,
        json={
            "concertId": 10,
            "title": "MT 뮤지컬",
            "price": 50000,
            "location": "가평",
            "notice": "MT 공연 안내",
            "casting": "홍길동",
            "detail": "MT 뮤지컬은 12월 5일 가평에서 진행됩니다.",
            "priceInfo": "일반석 50000원",
            "guide": "공연 시작 30분 전 입장",
            "refund": "취소는 공연 3일 전까지 가능합니다.",
            "story": "즐거운 MT 이야기",
            "mood": "밝은 분위기"
        }
    )

    assert r.status_code == 200
    assert r.json()["chunks"] >= 1

    # 공연 검색
    r = client.post(
        "/query",
        headers=HEADERS,
        json={
            "question": "MT 뮤지컬은 어디에서 진행돼?",
            "concertIds": [10],
            "topK": 4
        }
    )

    body = r.json()

    assert r.status_code == 200
    assert body["abstained"] is False
    assert body["sources"][0]["concertId"] == 10

    # 공연 색인 삭제
    r = client.delete(
        "/index/concert/10",
        headers=HEADERS
    )

    assert r.status_code == 200
    assert r.json()["deleted"] >= 1