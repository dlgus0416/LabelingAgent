"""
에이전트 라벨링 서비스 - Streamlit UI (v0.2)
실행: streamlit run app.py
"""

import streamlit as st
import json, os, re, time, io
import pandas as pd
import psycopg2
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# ====================================
# 페이지 설정
# ====================================
st.set_page_config(
    page_title="라벨링 에이전트",
    page_icon="🏷️",
    layout="wide",
)

# ====================================
# OpenAI 클라이언트 (세션 캐싱)
# ====================================
@st.cache_resource
def get_openai_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key)


GPT_MODEL = "gpt-4.1-mini"


# ====================================
# 공통 유틸리티
# ====================================
def parse_llm_json(text: str) -> Any:
    text = text.strip()
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def llm_call(client, system: str, user: str, temperature=0.3, max_tokens=3000) -> str:
    resp = client.responses.create(
        model=GPT_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    return resp.output_text.strip()


def llm_json(client, system: str, user: str, **kw) -> Any:
    return parse_llm_json(llm_call(client, system, user, **kw))


# ====================================
# 노드 함수들 (LabelingAgent.ipynb 에서 이식)
# ====================================

def node_collect_all_inputs(state, db_config=None, schemas=None, uploaded_files=None, log=None):
    """Node 1: 입력 수집"""
    log("입력 수집 시작...")
    sources = state.get("sources", [])

    # --- DB 정형 데이터 ---
    if db_config and schemas:
        try:
            conn = psycopg2.connect(**db_config)
            cur = conn.cursor()
            cur.execute("""
                SELECT t.table_schema, t.table_name,
                       c.column_name, c.data_type, c.ordinal_position
                FROM information_schema.tables t
                JOIN information_schema.columns c
                  ON t.table_schema = c.table_schema AND t.table_name = c.table_name
                WHERE t.table_schema = ANY(%s) AND t.table_type = 'BASE TABLE'
                ORDER BY t.table_schema, t.table_name, c.ordinal_position
            """, (schemas,))
            rows = cur.fetchall()
            cur.close()
            conn.close()

            tables = defaultdict(lambda: {"columns": []})
            for schema, tbl, col, dtype, pos in rows:
                key = f"{schema}.{tbl}"
                tables[key]["schema"] = schema
                tables[key]["table"] = tbl
                tables[key]["columns"].append({"name": col, "type": dtype, "position": pos})

            for key, tdata in tables.items():
                sources.append({
                    "source_id": f"src_{key}",
                    "source_type": "structured",
                    "payload": {
                        "table_name": tdata["table"],
                        "schema_name": tdata["schema"],
                        "columns": tdata["columns"],
                    },
                    "metadata": {"content_type": "db_table"},
                })
            log(f"DB 정형 소스: {len(tables)}개 테이블 수집")
        except Exception as e:
            log(f"DB 수집 실패: {e}")

    # --- 업로드 파일 (비정형/반정형) ---
    if uploaded_files:
        for f in uploaded_files:
            fname = f.name
            if fname.endswith((".xlsx", ".xls", ".csv")):
                try:
                    if fname.endswith(".csv"):
                        df = pd.read_csv(f)
                    else:
                        df = pd.read_excel(f)
                    sources.append({
                        "source_id": f"src_file_{fname}",
                        "source_type": "semi_structured",
                        "payload": {"json": df.to_dict(orient="records")},
                        "metadata": {"filename": fname, "content_type": "spreadsheet"},
                    })
                    log(f"파일 수집: {fname} ({len(df)}행)")
                except Exception as e:
                    log(f"파일 로드 실패 ({fname}): {e}")

            elif fname.endswith(".json"):
                try:
                    data = json.loads(f.read().decode("utf-8"))
                    sources.append({
                        "source_id": f"src_file_{fname}",
                        "source_type": "semi_structured",
                        "payload": {"json": data},
                        "metadata": {"filename": fname, "content_type": "application/json"},
                    })
                    log(f"파일 수집: {fname}")
                except Exception as e:
                    log(f"JSON 로드 실패 ({fname}): {e}")

            elif fname.endswith(".pdf"):
                try:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                    sources.append({
                        "source_id": f"src_file_{fname}",
                        "source_type": "unstructured",
                        "payload": {"text": text},
                        "metadata": {"filename": fname, "content_type": "application/pdf"},
                    })
                    log(f"PDF 수집: {fname} ({len(text)}자)")
                except ImportError:
                    log(f"PDF 처리에 PyPDF2 패키지 필요 (pip install PyPDF2)")
                except Exception as e:
                    log(f"PDF 로드 실패 ({fname}): {e}")

            elif fname.endswith(".txt"):
                text = f.read().decode("utf-8")
                sources.append({
                    "source_id": f"src_file_{fname}",
                    "source_type": "unstructured",
                    "payload": {"text": text},
                    "metadata": {"filename": fname, "content_type": "text/plain"},
                })
                log(f"텍스트 수집: {fname} ({len(text)}자)")

    state["sources"] = sources

    struct = sum(1 for s in sources if s["source_type"] == "structured")
    semi = sum(1 for s in sources if s["source_type"] == "semi_structured")
    unst = sum(1 for s in sources if s["source_type"] == "unstructured")
    log(f"총 {len(sources)}건 (정형 {struct}, 반정형 {semi}, 비정형 {unst})")
    return state


def node_structured_format_and_koreanize(state, client, log, detail=None):
    """Node 2: 정형 한글화"""
    log("정형 데이터 한글화...")
    struct_sources = [s for s in state.get("sources", []) if s["source_type"] == "structured"]
    if not struct_sources:
        state["structured_artifacts"] = []
        return state

    if detail:
        detail(f"정형 소스 {len(struct_sources)}개 테이블 처리 시작")

    schema_map = defaultdict(list)
    for s in struct_sources:
        schema_map[s["payload"]["schema_name"]].append(s["payload"]["table_name"])

    schema_descs = {}
    for schema_nm, tbl_list in schema_map.items():
        if detail:
            detail(f"스키마 분석 중: {schema_nm} ({len(tbl_list)}개 테이블)")
        try:
            desc = llm_call(client, "DB 구조 분석 전문가입니다.",
                f"스키마 '{schema_nm}' (테이블: {', '.join(tbl_list[:20])}) 의 비즈니스 도메인을 한 문장으로 설명하세요.",
                max_tokens=200)
            schema_descs[schema_nm] = desc
        except:
            schema_descs[schema_nm] = ""

    import threading
    _done_count = {"n": 0}
    _lock = threading.Lock()
    total = len(struct_sources)

    def _koreanize_one(source, schema_desc):
        p = source["payload"]
        col_list = "\n".join(f"- {c['name']} ({c['type']})" for c in p["columns"])
        try:
            result = llm_json(client, "DB 구조 분석 전문가입니다. JSON으로만 응답.",
                f"스키마: {p['schema_name']}\n스키마 설명: {schema_desc}\n테이블: {p['table_name']}\n컬럼:\n{col_list}\n\nJSON: {{\"table_comment\": \"설명\", \"columns\": [{{\"col_nm\": \"컬럼명\", \"col_comment\": \"설명\"}}]}}\n모든 {len(p['columns'])}개 컬럼 포함.",
                max_tokens=2000)
            with _lock:
                _done_count["n"] += 1
                if detail:
                    detail(f"한글화 진행: {_done_count['n']}/{total} — ✅ {p['schema_name']}.{p['table_name']} ({len(p['columns'])}컬럼)")
            return {"source_id": source["source_id"], "schema": p["schema_name"], "table": p["table_name"],
                    "table_comment": result.get("table_comment", ""), "columns_ko": result.get("columns", []),
                    "columns_raw": p["columns"], "schema_desc": schema_desc}
        except Exception as e:
            with _lock:
                _done_count["n"] += 1
                if detail:
                    detail(f"한글화 진행: {_done_count['n']}/{total} — ❌ {p['schema_name']}.{p['table_name']} (오류)")
            return {"source_id": source["source_id"], "schema": p["schema_name"], "table": p["table_name"],
                    "table_comment": f"오류: {e}", "columns_ko": [{"col_nm": c["name"], "col_comment": "실패"} for c in p["columns"]],
                    "columns_raw": p["columns"], "schema_desc": schema_desc}

    translated = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(_koreanize_one, src, schema_descs.get(src["payload"]["schema_name"], "")) for src in struct_sources]
        for f in futures:
            translated.append(f.result())

    artifacts = []
    for t in translated:
        artifacts.append({
            "source_id": t["source_id"],
            "normalized_table": {"table_name": t["table"], "schema_name": t["schema"],
                "columns": [{"name": c["name"], "dtype": c["type"]} for c in t["columns_raw"]]},
            "koreanized_schema": {"table_name_ko": t["table_comment"],
                "columns_ko": [{"name": c.get("col_nm",""), "name_ko": c.get("col_comment","")} for c in t["columns_ko"]]},
        })
    state["structured_artifacts"] = artifacts
    log(f"한글화 완료: {len(artifacts)}건")
    return state


def node_structured_doc_builder(state, client, db_config, log, detail=None):
    """Node 3: Table Doc 생성"""
    log("Table Doc 생성...")
    artifacts = state.get("structured_artifacts", [])
    if not artifacts:
        state["table_docs"] = []
        return state

    if detail:
        detail(f"Table Doc {len(artifacts)}건 생성 시작")

    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        db_ok = True
    except:
        db_ok = False

    table_docs = []
    for idx, art in enumerate(artifacts):
        nt, ks = art["normalized_table"], art["koreanized_schema"]
        if detail:
            detail(f"Table Doc 진행: {idx+1}/{len(artifacts)} — {nt['schema_name']}.{nt['table_name']}")
        cols_text = "\n".join(f"- {r['name']} ({r['dtype']}): {k.get('name_ko','')}" for r, k in zip(nt["columns"], ks["columns_ko"]))

        sample_text = "(없음)"
        if db_ok:
            try:
                cur.execute(f"SELECT * FROM {nt['schema_name']}.{nt['table_name']} LIMIT 5")
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                sample_text = json.dumps(rows[:3], ensure_ascii=False, indent=2, default=str)
            except:
                pass

        try:
            doc_md = llm_call(client,
                "테이블 분석 전문가. 섹션: ### 테이블 정의, ### 컬럼 정의, ### 인사이트 (3~5개), ### 예상 질문 (5~10개)",
                f"스키마: {nt['schema_name']}\n테이블: {nt['table_name']} ({ks['table_name_ko']})\n컬럼:\n{cols_text}\n샘플:\n{sample_text[:1500]}",
                max_tokens=2000)
            table_docs.append({"source_id": art["source_id"], "doc_md": f"# {nt['table_name']} ({ks['table_name_ko']})\n\n{doc_md}"})
        except Exception as e:
            table_docs.append({"source_id": art["source_id"], "doc_md": f"# {nt['table_name']}\n\n오류: {e}"})

    if db_ok:
        cur.close()
        conn.close()

    state["table_docs"] = table_docs
    log(f"Table Doc 생성 완료: {len(table_docs)}건")
    return state


def node_chunk(state, log):
    """Node 4: Chunking"""
    log("Chunking...")
    non_struct = [s for s in state.get("sources", []) if s["source_type"] != "structured"]
    chunks = []
    idx = 0
    for src in non_struct:
        sid, stype, payload = src["source_id"], src["source_type"], src["payload"]
        if stype == "semi_structured":
            json_data = payload.get("json", payload)
            items = json_data if isinstance(json_data, list) else json_data.get("events", json_data.get("items", [json_data]))
            if not isinstance(items, list):
                items = [items]
            for i, item in enumerate(items):
                chunks.append({"chunk_id": f"chk_{idx:04d}", "source_id": sid, "source_type": stype,
                               "target_ref": f"event:{sid}.idx:{i}", "content": item})
                idx += 1
        elif stype == "unstructured":
            text = payload.get("text", "")
            cs, ov = 500, 100
            if len(text) <= cs:
                chunks.append({"chunk_id": f"chk_{idx:04d}", "source_id": sid, "source_type": stype,
                               "target_ref": f"doc:{sid}.span:0-{len(text)}", "content": text})
                idx += 1
            else:
                start = 0
                while start < len(text):
                    end = min(start + cs, len(text))
                    chunks.append({"chunk_id": f"chk_{idx:04d}", "source_id": sid, "source_type": stype,
                                   "target_ref": f"doc:{sid}.span:{start}-{end}", "content": text[start:end]})
                    idx += 1
                    start += cs - ov
    state["chunks"] = chunks
    log(f"Chunking 완료: {len(chunks)}건")
    return state


def node_source_doc_builder(state, client, log):
    """Node 5: Source Doc 생성"""
    log("Source Doc 생성...")
    non_struct = [s for s in state.get("sources", []) if s["source_type"] != "structured"]
    chunks = state.get("chunks", [])
    source_docs = []
    for src in non_struct:
        sid, stype = src["source_id"], src["source_type"]
        src_chunks = [c for c in chunks if c["source_id"] == sid]
        preview = []
        for c in src_chunks[:10]:
            ct = c["content"] if isinstance(c["content"], str) else json.dumps(c["content"], ensure_ascii=False)
            preview.append(ct[:300])
        try:
            result = llm_json(client,
                "데이터 분석 전문가. JSON으로만 응답.",
                f"source_id: {sid}\nsource_type: {stype}\n청크 수: {len(src_chunks)}\n미리보기:\n" + "\n".join(preview) +
                f"\n\nJSON: {{\"source_summary\": [\"...\"], \"expected_questions\": [\"...\"]}}\nsource_summary: 5~10줄. expected_questions: 10~20개.",
                max_tokens=2000)
            source_docs.append({"source_id": sid, "source_type": stype,
                "source_summary": result.get("source_summary", []), "expected_questions": result.get("expected_questions", [])})
        except Exception as e:
            source_docs.append({"source_id": sid, "source_type": stype, "source_summary": [f"오류: {e}"], "expected_questions": []})
    state["source_docs"] = source_docs
    log(f"Source Doc 완료: {len(source_docs)}건")
    return state


def node_biz_word_agent(state, client, log):
    """Node 7: 라벨 후보 추출"""
    log("라벨 후보 추출...")
    ctx = ""
    for td in state.get("table_docs", [])[:5]:
        ctx += f"[{td['source_id']}]\n{td['doc_md'][:500]}\n\n"
    for sd in state.get("source_docs", [])[:5]:
        ctx += f"[{sd['source_id']}] " + " ".join(sd.get("source_summary", [])[:3]) + "\n"
    for c in state.get("chunks", [])[:10]:
        ct = c["content"] if isinstance(c["content"], str) else json.dumps(c["content"], ensure_ascii=False)
        ctx += f"[{c['chunk_id']}] {ct[:200]}\n"

    try:
        result = llm_json(client,
            "biz_word_agent. 데이터에서 비즈니스 용어 후보 추출. observed_aliases/suggested_aliases 분리. JSON으로만 응답.",
            f"{ctx[:4000]}\n\nJSON: {{\"label_candidates\": [{{\"candidate_id\": \"cand_0001\", \"label\": \"라벨명\", \"definition\": \"정의\", \"category\": \"분류\", \"aliases\": [], \"observed_aliases\": [], \"suggested_aliases\": [], \"evidence\": [{{\"source_id\": \"...\", \"snippet\": \"...\", \"confidence\": 0.9}}]}}]}}\n10~30개 추출.",
            max_tokens=4000)
        cands = result.get("label_candidates", [])
        for i, c in enumerate(cands):
            if not c.get("candidate_id"):
                c["candidate_id"] = f"cand_{i+1:04d}"
            c["aliases"] = list(set(c.get("aliases", []) + c.get("observed_aliases", []) + c.get("suggested_aliases", [])))
        state["label_candidates"] = cands
        state["label_candidates_version"] = f"v1_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        log(f"후보 추출 완료: {len(cands)}개")
    except Exception as e:
        log(f"후보 추출 실패: {e}")
        state["label_candidates"] = []
    return state


def node_planner_agent(state, client, log):
    """Node 6: 가이드라인 생성"""
    log("가이드라인 생성...")
    cands = state.get("label_candidates", [])
    if not cands:
        state["alias_map"] = {}
        state["labeling_guideline_md"] = ""
        state["label_schema"] = {}
        return state

    cands_text = json.dumps([{"label": c["label"], "definition": c.get("definition",""), "aliases": c.get("aliases",[])} for c in cands[:30]], ensure_ascii=False, indent=2)
    try:
        result = llm_json(client,
            "planner_agent. alias_map + 라벨링 가이드라인 생성. JSON으로만 응답.",
            f"라벨 후보:\n{cands_text}\n\nJSON: {{\"alias_map\": {{\"유의어\": \"정규라벨\"}}, \"labeling_guideline_md\": \"# 가이드라인\\n...\", \"label_schema\": {{\"type\": \"object\", \"properties\": {{\"label\": {{\"type\": \"string\"}}, \"matched_alias\": {{\"type\": \"string\"}}, \"confidence\": {{\"type\": \"number\"}}, \"rationale\": {{\"type\": \"string\"}}}}, \"required\": [\"label\"]}}}}",
            max_tokens=4000)
        state["alias_map"] = result.get("alias_map", {})
        state["labeling_guideline_md"] = result.get("labeling_guideline_md", "")
        state["label_schema"] = result.get("label_schema", {})
        log(f"alias_map: {len(state['alias_map'])}개, guideline: {len(state['labeling_guideline_md'])}자")
    except Exception as e:
        log(f"가이드라인 생성 실패: {e}")
        alias_map = {}
        for c in cands:
            for a in c.get("aliases", []):
                if a != c["label"]:
                    alias_map[a] = c["label"]
        state["alias_map"] = alias_map
        state["labeling_guideline_md"] = "# 기본 규칙 적용"
        state["label_schema"] = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}
    return state


def node_labeler_agent(state, client, log):
    """Node 8: 라벨 부착"""
    log("라벨 부착...")
    gmd = state.get("labeling_guideline_md", "")
    am = state.get("alias_map", {})
    cands = state.get("label_candidates", [])
    if not cands:
        state["labeled_items"] = []
        return state

    valid = {c["label"] for c in cands}
    ct = "\n".join(f"- {c['label']}: {c.get('definition','')} (유의어: {', '.join(c.get('aliases',[]))})" for c in cands)

    def _batch(items_json, context):
        try:
            result = llm_json(client,
                "labeler_agent. 라벨 후보만 사용. 유의어→alias_map 매핑. JSON으로만 응답.",
                f"가이드라인:\n{gmd[:2000]}\nalias_map: {json.dumps(am, ensure_ascii=False)}\n라벨 후보:\n{ct}\n컨텍스트:\n{context[:1500]}\n대상:\n{items_json}\n\nJSON: {{\"labeled_items\": [{{\"source_id\": \"...\", \"target_ref\": \"...\", \"labels\": [{{\"label\": \"정규라벨\", \"matched_alias\": null, \"confidence\": 0.9, \"rationale\": \"...\", \"evidence\": [\"...\"]}}]}}]}}",
                max_tokens=4000)
            labeled = []
            for item in result.get("labeled_items", []):
                vl = []
                for lbl in item.get("labels", []):
                    n = lbl.get("label", "")
                    if n in valid:
                        vl.append(lbl)
                    else:
                        c2 = am.get(n)
                        if c2 and c2 in valid:
                            lbl["matched_alias"] = n
                            lbl["label"] = c2
                            vl.append(lbl)
                labeled.append({"source_id": item.get("source_id",""), "target_ref": item.get("target_ref",""), "labels": vl})
            return labeled
        except:
            return []

    all_labeled = []

    for doc in state.get("table_docs", []):
        ij = json.dumps([{"source_id": doc["source_id"], "target_ref": f"table:{doc['source_id']}", "content": doc["doc_md"][:2000]}], ensure_ascii=False)
        all_labeled.extend(_batch(ij, doc["doc_md"][:1000]))

    chunks = state.get("chunks", [])
    sd_map = {sd["source_id"]: sd for sd in state.get("source_docs", [])}
    for i in range(0, len(chunks), 10):
        batch = chunks[i:i+10]
        context = ""
        for sid in {c["source_id"] for c in batch}:
            sd = sd_map.get(sid)
            if sd:
                context += " ".join(sd.get("source_summary", [])[:3]) + "\n"
        items = [{"source_id": c["source_id"], "target_ref": c["target_ref"], "chunk_id": c["chunk_id"],
                  "content": (c["content"] if isinstance(c["content"], str) else json.dumps(c["content"], ensure_ascii=False))[:500]} for c in batch]
        all_labeled.extend(_batch(json.dumps(items, ensure_ascii=False), context))

    state["labeled_items"] = all_labeled
    total = sum(len(x.get("labels", [])) for x in all_labeled)
    log(f"라벨 부착 완료: {len(all_labeled)}건, 라벨 {total}개")
    return state


def node_reviewer_agent(state, client, log):
    """Node 9: 리뷰"""
    log("라벨 검토...")
    labeled = state.get("labeled_items", [])
    items_to_review = [i for i in labeled if i.get("labels")]
    if not items_to_review:
        state["review_results"] = []
        state["needs_human"] = False
        return state

    try:
        result = llm_json(client,
            "reviewer_agent. alias_map 일치/근거 충분성/다의어 검증. JSON으로만 응답.",
            f"가이드라인:\n{state.get('labeling_guideline_md','')[:1500]}\nalias_map: {json.dumps(state.get('alias_map',{}), ensure_ascii=False)}\n검토 대상:\n{json.dumps(items_to_review[:20], ensure_ascii=False, indent=2)}\n\nJSON: {{\"review_results\": [{{\"source_id\": \"...\", \"target_ref\": \"...\", \"label\": \"...\", \"verdict\": \"pass/fail\", \"reason\": \"...\", \"suggested_fix\": null}}]}}",
            max_tokens=4000)
        reviews = result.get("review_results", [])
        state["review_results"] = reviews
        fail_ct = sum(1 for r in reviews if r.get("verdict") == "fail")
        state["needs_human"] = fail_ct > 0
        log(f"검토 완료: pass {len(reviews)-fail_ct}, fail {fail_ct}")
    except Exception as e:
        log(f"검토 실패: {e}")
        state["review_results"] = []
        state["needs_human"] = False
    return state


def node_hitl(state, log):
    """Node 10: HITL"""
    log("HITL 처리...")
    reviews = state.get("review_results", [])
    fail_keys = {(r.get("source_id"), r.get("target_ref"), r.get("label")) for r in reviews if r.get("verdict") == "fail"}
    hitl_decisions = []
    final_items = []
    for item in state.get("labeled_items", []):
        kept = []
        for lbl in item.get("labels", []):
            key = (item.get("source_id"), item.get("target_ref"), lbl.get("label"))
            if key in fail_keys:
                hitl_decisions.append({"item_ref": f"{key[0]}|{key[1]}|{key[2]}", "action": "remove"})
            else:
                kept.append(lbl)
        if kept:
            final_items.append({"source_id": item["source_id"], "target_ref": item["target_ref"], "labels": kept})
    state["hitl_decisions"] = hitl_decisions
    state["final_labeled_items"] = final_items
    log(f"HITL: {len(hitl_decisions)}건 처리, 최종 {len(final_items)}건")
    return state


def node_save(state, log):
    """Node 11: 저장"""
    log("결과 저장...")
    final = state.get("final_labeled_items") or state.get("labeled_items", [])
    rows = []
    for item in final:
        for lbl in item.get("labels", []):
            rows.append({
                "source_id": item["source_id"],
                "target_ref": item["target_ref"],
                "label": lbl.get("label", ""),
                "matched_alias": lbl.get("matched_alias"),
                "confidence": lbl.get("confidence"),
                "rationale": lbl.get("rationale", ""),
                "evidence": json.dumps(lbl.get("evidence", []), ensure_ascii=False),
            })
    state["result_df"] = pd.DataFrame(rows)
    state["db_write_result"] = {
        "sources": len(state.get("sources", [])),
        "documents": len(state.get("table_docs", [])) + len(state.get("source_docs", [])),
        "chunks": len(state.get("chunks", [])),
        "candidates": len(state.get("label_candidates", [])),
        "labels": len(rows),
        "reviews": len(state.get("review_results", [])),
        "hitl": len(state.get("hitl_decisions", [])),
    }
    log(f"저장 완료: {len(rows)}개 라벨")
    return state


# ====================================
# Streamlit UI
# ====================================

st.title("🏷️ 에이전트 라벨링 서비스")
st.caption("v0.2 — 정형/반정형/비정형 데이터를 입력하여 자동 라벨링 워크플로우를 실행합니다.")

# --- 사이드바: API 키 ---
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("OpenAI API Key",
        value=os.environ.get("OPENAI_API_KEY", "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"),
        type="password")

# --- 데이터 소스 선택 ---
st.header("1. 데이터 소스 선택")

col_check1, col_check2 = st.columns(2)
with col_check1:
    use_db = st.checkbox("📊 DB (정형 데이터)", value=True)
with col_check2:
    use_files = st.checkbox("📁 파일 업로드 (PDF / Excel / JSON / TXT)")

# --- DB 설정 ---
db_config = None
schemas = None

if use_db:
    st.subheader("DB 연결 정보")
    col_db1, col_db2 = st.columns(2)
    with col_db1:
        db_host = st.text_input("Host", value="yspark-enai.c2n061ax1rzb.ap-northeast-2.rds.amazonaws.com")
        db_name = st.text_input("Database", value="dbt")
        db_user = st.text_input("User", value="spider2")
    with col_db2:
        db_port = st.text_input("Port", value="15432")
        db_pw = st.text_input("Password", value="!Encore2026", type="password")

    db_config = {"host": db_host, "dbname": db_name, "user": db_user, "password": db_pw, "port": db_port}

    # DB 연결 테스트 + 스키마 목록 조회
    if st.button("🔌 DB 연결 테스트"):
        try:
            conn = psycopg2.connect(**db_config)
            cur = conn.cursor()
            cur.execute("""
                SELECT schema_name FROM information_schema.schemata
                WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                ORDER BY schema_name
            """)
            available_schemas = [row[0] for row in cur.fetchall()]
            cur.close()
            conn.close()
            st.success(f"연결 성공! {len(available_schemas)}개 스키마 발견")
            st.session_state["available_schemas"] = available_schemas
        except Exception as e:
            st.error(f"연결 실패: {e}")

    # 스키마 선택
    available = st.session_state.get("available_schemas", [
        "spider2_playbook001", "spider2_google_play001",
        "spider2_activity001", "spider2_airport001", "spider2_airbnb001"
    ])
    schemas = st.multiselect("처리할 스키마 선택", options=available,
        default=[s for s in ["spider2_activity001"] if s in available])

# --- 파일 업로드 ---
uploaded_files = None
if use_files:
    st.subheader("파일 업로드")
    uploaded_files = st.file_uploader(
        "파일을 드래그 & 드롭하거나 선택하세요",
        type=["pdf", "xlsx", "xls", "csv", "json", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.info(f"{len(uploaded_files)}개 파일 선택됨: {', '.join(f.name for f in uploaded_files)}")

# --- 실행 ---
st.header("2. 워크플로우 실행")

can_run = (use_db and db_config and schemas) or (use_files and uploaded_files)

if st.button("🚀 라벨링 워크플로우 실행", disabled=not can_run, type="primary"):
    client = get_openai_client(api_key)

    log_area = st.empty()
    logs = []

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        logs.append(f"`{ts}` {msg}")
        log_area.markdown("\n\n".join(logs))

    state = {
        "request_id": f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "sources": [],
    }

    progress = st.progress(0, text="워크플로우 시작...")
    steps = [
        ("입력 수집", lambda s: node_collect_all_inputs(s, db_config if use_db else None, schemas, uploaded_files, log)),
    ]

    # 동적으로 경로 결정 (수집 후)
    total_steps = 9  # 최대 단계 수

    # Step 1: collect
    progress.progress(1/total_steps, text="Step 1/9: 입력 수집")
    state = node_collect_all_inputs(state, db_config if use_db else None, schemas, uploaded_files, log)

    has_struct = any(s["source_type"] == "structured" for s in state.get("sources", []))
    has_non = any(s["source_type"] != "structured" for s in state.get("sources", []))

    step = 2

    # Step 2-3: 정형 경로
    if has_struct:
        progress.progress(step/total_steps, text=f"Step {step}/9: 정형 한글화")
        state = node_structured_format_and_koreanize(state, client, log)
        step += 1

        progress.progress(step/total_steps, text=f"Step {step}/9: Table Doc 생성")
        state = node_structured_doc_builder(state, client, db_config or {}, log)
        step += 1

    # Step 2-3 alt: 비정형 경로
    if has_non:
        progress.progress(step/total_steps, text=f"Step {step}/9: Chunking")
        state = node_chunk(state, log)
        step += 1

        progress.progress(step/total_steps, text=f"Step {step}/9: Source Doc 생성")
        state = node_source_doc_builder(state, client, log)
        step += 1

    # Step 4: biz_word
    progress.progress(min(step/total_steps, 0.9), text=f"Step {step}/9: 라벨 후보 추출")
    state = node_biz_word_agent(state, client, log)
    step += 1

    # Step 5: planner
    progress.progress(min(step/total_steps, 0.9), text=f"Step {step}/9: 가이드라인 생성")
    state = node_planner_agent(state, client, log)
    step += 1

    # Step 6: labeler
    progress.progress(min(step/total_steps, 0.9), text=f"Step {step}/9: 라벨 부착")
    state = node_labeler_agent(state, client, log)
    step += 1

    # Step 7: reviewer
    progress.progress(min(step/total_steps, 0.95), text=f"Step {step}/9: 라벨 검토")
    state = node_reviewer_agent(state, client, log)
    step += 1

    # Step 8: HITL (조건부)
    if state.get("needs_human"):
        state = node_hitl(state, log)

    # Step 9: save
    progress.progress(0.99, text="Step 9/9: 결과 저장")
    state = node_save(state, log)

    progress.progress(1.0, text="완료!")
    st.session_state["result_state"] = state
    st.success("워크플로우 완료!")

# --- 결과 표시 ---
if "result_state" in st.session_state:
    state = st.session_state["result_state"]
    st.header("3. 결과")

    # 요약 메트릭
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sources", len(state.get("sources", [])))
    col2.metric("라벨 후보", len(state.get("label_candidates", [])))
    col3.metric("부착 라벨", sum(len(x.get("labels",[])) for x in (state.get("final_labeled_items") or state.get("labeled_items", []))))
    col4.metric("리뷰 Fail", sum(1 for r in state.get("review_results", []) if r.get("verdict") == "fail"))

    # 탭
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 라벨 결과", "🏷️ 라벨 후보", "📋 가이드라인", "🔍 리뷰 결과", "📄 alias_map"])

    with tab1:
        df = state.get("result_df")
        if df is not None and len(df) > 0:
            st.dataframe(df, use_container_width=True)

            # 다운로드
            buf = io.BytesIO()
            df.to_excel(buf, index=False, engine="openpyxl")
            st.download_button("📥 Excel 다운로드", data=buf.getvalue(),
                file_name=f"labeling_results_{state.get('request_id','')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.info("라벨 결과 없음")

    with tab2:
        cands = state.get("label_candidates", [])
        if cands:
            df_c = pd.DataFrame([{
                "label": c["label"],
                "definition": c.get("definition", ""),
                "category": c.get("category", ""),
                "aliases": ", ".join(c.get("aliases", [])),
            } for c in cands])
            st.dataframe(df_c, use_container_width=True)
        else:
            st.info("라벨 후보 없음")

    with tab3:
        gmd = state.get("labeling_guideline_md", "")
        if gmd:
            st.markdown(gmd)
        else:
            st.info("가이드라인 없음")

    with tab4:
        reviews = state.get("review_results", [])
        if reviews:
            df_r = pd.DataFrame(reviews)
            st.dataframe(df_r, use_container_width=True)
        else:
            st.info("리뷰 결과 없음")

    with tab5:
        am = state.get("alias_map", {})
        if am:
            df_a = pd.DataFrame([{"유의어": k, "정규 라벨": v} for k, v in am.items()])
            st.dataframe(df_a, use_container_width=True)
        else:
            st.info("alias_map 없음")

    # 저장 요약
    with st.expander("저장 요약"):
        st.json(state.get("db_write_result", {}))
