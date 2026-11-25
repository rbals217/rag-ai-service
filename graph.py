"""LangGraph RAG 그래프: retrieve → grade → generate | rewrite↺ | abstain → format"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from engine import ABSTAIN_MESSAGE, Chunk, body_of
from settings import Settings


class RagState(TypedDict, total=False):
    question: str
    query: str
    concert_ids: list[int]
    top_k: int
    chunks: list[Chunk]
    attempts: int
    answer: str
    abstained: bool
    sources: list[dict]
    trace: Annotated[list[str], operator.add]


def build_graph(engine, settings: Settings):
    """공연 검색 및 답변 생성을 위한 LangGraph를 구성한다."""

    def retrieve(state: RagState) -> dict:
        concert_ids = state.get("concert_ids") or []

        chunks = engine.search(
            state["query"],
            concert_ids,
            state.get("top_k", 4)
        )

        return {"chunks": chunks, "trace": ["retrieve"]}

    def grade(state: RagState) -> dict:
        kept = [
            chunk
            for chunk in state.get("chunks", [])
            if chunk.score >= engine.min_score
        ]
        return {"chunks": kept, "trace": ["grade"]}

    def route_after_grade(state: RagState) -> str:
        if state.get("chunks"):
            return "generate"

        # settings.py에 max_rewrites가 없을 때는 기본 1회 재작성
        max_rewrites = getattr(settings, "max_rewrites", 1)
        if (
            state.get("concert_ids")
            and state.get("attempts", 0) < max_rewrites
        ):
            return "rewrite"

        return "abstain"

    def rewrite(state: RagState) -> dict:
        return {
            "query": engine.rewrite(state["question"]),
            "attempts": state.get("attempts", 0) + 1,
            "trace": ["rewrite"],
        }

    def generate(state: RagState) -> dict:
        return {
            "answer": engine.generate(state["question"], state["chunks"]),
            "abstained": False,
            "trace": ["generate"],
        }

    def abstain(state: RagState) -> dict:
        return {
            "answer": ABSTAIN_MESSAGE,
            "abstained": True,
            "chunks": [],
            "trace": ["abstain"],
        }

    def format_(state: RagState) -> dict:
        # 같은 공연의 여러 청크 중 점수가 가장 높은 청크 하나만 출처로 표시
        best: dict[int, Chunk] = {}

        for chunk in state.get("chunks", []):
            concert_id = chunk.concert_id
            if (
                concert_id not in best
                or chunk.score > best[concert_id].score
            ):
                best[concert_id] = chunk

        sources = [
            {
                "concert_id": chunk.concert_id,
                "title": chunk.title,
                "field_name": chunk.field_name,
                "snippet": body_of(chunk.text)[:120],
                "score": chunk.score,
            }
            for chunk in sorted(
                best.values(),
                key=lambda item: item.score,
                reverse=True,
            )
        ]
        return {"sources": sources, "trace": ["format"]}

    graph = StateGraph(RagState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.add_node("abstain", abstain)
    graph.add_node("format", format_)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "abstain": "abstain",
        },
    )

    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "format")
    graph.add_edge("abstain", "format")
    graph.add_edge("format", END)

    return graph.compile()


def run_graph(
    graph,
    question: str,
    concert_ids: list[int],
    top_k: int,
) -> dict:
    """그래프를 실행하고 최종 상태를 반환한다."""
    return graph.invoke(
        {
            "question": question,
            "query": question,
            "concert_ids": concert_ids,
            "top_k": top_k,
            "attempts": 0,
            "trace": [],
        }
    )
