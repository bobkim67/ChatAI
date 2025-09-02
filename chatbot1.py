# %%
# OPENAI_API_KEY=sk-proj
import pandas as pd
import streamlit as st
from openai import OpenAI
import pnsn_sim_calculator
from datetime import date as _date
import os, re
import requests
from io import BytesIO
from urllib.parse import urlsplit, urlunsplit, quote

def style_dataframe(df: pd.DataFrame):
    fmt = {}

    # float만 포맷팅
    for col in df.select_dtypes(include="float"):
        if col == "세율":
            fmt[col] = "{:.1%}"
        else:
            fmt[col] = "{:,.0f}"  # 소수 둘째자리까지 + 천 단위 구분

    # int만 포맷팅
    for col in df.select_dtypes(include="int"):
        fmt[col] = "{:,.0f}"

    return df.style.format(fmt)

def _to_raw_if_github(url: str) -> str:
    # github blob → raw
    return re.sub(
        r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)",
        r"https://raw.githubusercontent.com/\1/\2/\3/\4",
        url,
    )

def _percent_encode_path(url: str) -> str:
    s = urlsplit(url)
    # path만 퍼센트 인코딩 (한글/공백/특수문자 대응)
    return urlunsplit((s.scheme, s.netloc, quote(s.path), s.query, s.fragment))

def read_excel_safely(src: str, *, sheet_name=0, github_token: str | None = None):
    # 1) 로컬 파일이면 바로 읽기
    if not src.lower().startswith(("http://", "https://")):
        return pd.read_excel(src, sheet_name=sheet_name)

    # 2) 깃허브 blob → raw, 경로 인코딩
    url = _percent_encode_path(_to_raw_if_github(src))

    # 3) 헤더 구성 (일부 서버는 UA 없으면 403)
    headers = {"User-Agent": "Mozilla/5.0"}
    # 프라이빗 repo면 토큰 사용 (Streamlit에선 st.secrets 권장)
    token = github_token or os.environ.get("GITHUB_TOKEN")
    if token and ("api.github.com" in url or "raw.githubusercontent.com" in url):
        headers["Authorization"] = f"token {token}"

    # 4) 요청 & 예외 처리
    r = requests.get(url, headers=headers, timeout=30)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # 깃허브 파일 뷰 URL을 그대로 넣었을 때 대비: ?raw=1 붙여 재시도
        if "github.com/" in src and "/blob/" in src:
            alt = _percent_encode_path(src.replace("github.com/", "github.com/") + "?raw=1")
            r = requests.get(_to_raw_if_github(alt), headers=headers, timeout=30)
            r.raise_for_status()
        else:
            raise RuntimeError(f"HTTPError {r.status_code} for URL: {url}") from e

    # 5) 바이트를 엑셀 파서로
    return pd.read_excel(BytesIO(r.content), sheet_name=sheet_name)

# 사용 예시 ----------------------------------------------------
# 모든 시트 읽기
dfs = read_excel_safely(
    "https://raw.githubusercontent.com/bobkim67/ChatAI/main/dataset_kor.xlsx",
    sheet_name=None,
    # github_token=st.secrets.get("GITHUB_TOKEN")  # 프라이빗이면 주석 해제
)
# 엑셀 파일 전체 시트 읽기
# dfs = pd.read_excel(file_path, sheet_name=None) 

# 시트명을 변수로 할당
계좌정보 = dfs["계좌정보"]
고객통합기본 = dfs["고객통합기본"]
퇴직금통산급여계좌 = dfs["퇴직금통산급여계좌"]
DC가입자계약기본 = dfs["DC가입자계약기본"]
IRP계좌원천징수내역 = dfs["IRP계좌원천징수내역"]
IRP소득공제납입내역 = dfs["IRP소득공제납입내역"]
연금저축계좌기본 = dfs["연금저축계좌기본"]
연금저축계좌지급내역상세 = dfs["연금저축계좌지급내역상세"]
개인연금계약기본 = dfs["개인연금계약기본"]
 
all_ids = 고객통합기본["고객번호"].unique()
# %%
# 페이지 레이아웃을 넓게
st.set_page_config(page_title="Chatbot in Sidebar", layout="wide")

# ----(0) API 키 입력 (사이드바 상단)----
with st.sidebar:
    st.title("💬 Chatbot")
    api_key = st.text_input("OpenAI API Key", type="password")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-5"], index=0)

# if not api_key:
#     st.info("오른쪽 사이드바에 OpenAI API Key를 입력하세요.", icon="🗝️")
#     st.stop()

client = OpenAI(api_key=api_key, timeout=30, max_retries=3)

# ----(1) 세션 상태 초기화----
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user"/"assistant","content": "..."}]

# ----(2) 메인 영역: 여러분의 대시보드 / 차트 등----
# 고객번호 선택 위젯
selected_id = st.selectbox("고객번호 선택", sorted(all_ids))
# 탭 생성
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "고객정보",
        "개인형IRP",
        "DC",
        "연금저축계좌",
        "(구)개인연금",
        "연금시뮬레이션",
    ]
)
with tab1:
    st.markdown("##### 고객정보")
    st.dataframe(고객통합기본[고객통합기본["고객번호"] == selected_id])
    st.markdown("##### 계좌현황")
    st.dataframe(계좌정보[계좌정보["고객번호"] == selected_id])
    st.markdown("##### 퇴직금 수령이력")
    st.dataframe(퇴직금통산급여계좌[퇴직금통산급여계좌["고객번호"] == selected_id])

with tab2:
    st.markdown("##### IRP 소득공제 납입내역")
    st.dataframe(IRP소득공제납입내역[IRP소득공제납입내역["고객번호"] == selected_id])
    st.markdown("##### IRP 지급내역")
    st.dataframe(IRP계좌원천징수내역[IRP계좌원천징수내역["고객번호"] == selected_id])

with tab3:
    st.markdown("##### DC 계약현황")
    st.dataframe(DC가입자계약기본[DC가입자계약기본["고객번호"] == selected_id]) 

with tab4:
    st.markdown("##### 연금저축 계약현황")
    st.dataframe(연금저축계좌기본[연금저축계좌기본["고객번호"] == selected_id])
    st.markdown("##### 연금저축 지급내역")
    st.dataframe(
        연금저축계좌지급내역상세[연금저축계좌지급내역상세["고객번호"] == selected_id]
    )
with tab5:
    st.markdown("##### 개연연금 계약현황") 
    st.dataframe(개인연금계약기본[개인연금계약기본["고객번호"] == selected_id])

with tab6:
    st.markdown("##### 연금수령 시뮬레이션")
    if "지급기간_년" not in st.session_state:
        st.session_state["지급기간_년"] = 10  # <- 원하는 디폴트 

    # ★ 기본 디폴트 값 (요청값 반영)
    _def_평가기준일 = _date(2025, 9, 1)
    _def_생년월일   = _date(1968, 2, 15)
    _def_입사일     = _date(2009,10, 1)
    _def_퇴직일     = _date(2025, 9, 1)
    _def_IRP가입일   = _date(2014, 5, 1)
    _def_제도가입일  = _date(2014, 1, 1)
    _def_연금개시일  = _date(2025, 9, 1)  # 퇴직일 이후

    _def_과세제외_자기부담금   = 30_000_000
    _def_이연퇴직소득         = 500_000_000
    _def_세액공제자기부담금   = 150_000_000
    _def_운용손익             = 20_000_000
    _def_그외                 = _def_세액공제자기부담금 + _def_운용손익  # 170,000,000
    _def_총평가금액           = 700_000_000
    _def_운용수익률           = 0.03

    # 파생 디폴트: 개시 나이, 근속년수
    _def_수령나이 = (_def_연금개시일.year - _def_생년월일.year) - \
                    (1 if (_def_연금개시일.month, _def_연금개시일.day) < (_def_생년월일.month, _def_생년월일.day) else 0)    
    _def_근속년수 = int((_def_퇴직일 - _def_입사일).days // 365)
    _def_현재나이 = (_def_평가기준일.year - _def_생년월일.year) - \
                    (1 if (_def_평가기준일.month, _def_평가기준일.day) < (_def_생년월일.month, _def_생년월일.day) else 0)        

    with st.form("pension_inputs"):
        st.subheader("기본 정보(날짜)")
        d1, d2, d3 = st.columns(3)
        with d1:
            평가기준일 = st.date_input("평가기준일", value=_def_평가기준일)
            생년월일     = st.date_input("생년월일",   value=_def_생년월일)
            입사일       = st.date_input("입사일",     value=_def_입사일)
        with d2:
            퇴직일       = st.date_input("퇴직일",     value=_def_퇴직일)
            퇴직연금제도가입일 = st.date_input("퇴직연금 제도가입일", value=_def_제도가입일)
            IRP가입일    = st.date_input("IRP 가입일", value=_def_IRP가입일)

        # 🛠️ d1/d2에서 받은 값으로 '연금수령가능일' 즉시 산출
        _연금수령가능일_dt = pnsn_sim_calculator.calc_연금수령가능일(
            생년월일=생년월일, IRP가입일=IRP가입일, 퇴직일=퇴직일
        )

        with d3:
            # 🛠️ 산출된 '연금수령가능일'을 보여주기(읽기전용)
            st.date_input("연금수령가능일 (자동 계산)", value=_연금수령가능일_dt, disabled=True,
                          help="퇴직일, 55세 되는 날, IRP 가입일 + 5년 중 가장 늦은 날")

            # 개시일은 사용자가 선택(기본값은 디폴트와 자동 산출값 중 더 늦은 날로 제안)
            _개시_디폴트 = _연금수령가능일_dt if _연금수령가능일_dt > _def_연금개시일 else _def_연금개시일
            연금개시일   = st.date_input("연금개시일(연금수령가능일 이후)", value=_개시_디폴트)
            운용수익률   = st.number_input("연 운용수익률(예: 0.03=3%)", value=_def_운용수익률, step=0.005, format="%.3f")
        
        b1, b2, b3 = st.columns(3)
        with b1:
            # 개시 나이(사용자 조정 가능)
            _auto_수령나이 = (연금개시일.year - 생년월일.year) - \
                             (1 if (연금개시일.month, 연금개시일.day) < (생년월일.month, 생년월일.day) else 0)
            st.caption("연금개시 연령: " f"{_auto_수령나이}세")
        with b2:
            if 퇴직일 is not None and 입사일 is not None:
                근속월수 = (퇴직일.year - 입사일.year) * 12 + (퇴직일.month - 입사일.month)
                if 퇴직일.day < 입사일.day:
                    근속월수 -= 1
                _auto_근속년수 = math.ceil((근속월수 + 1) / 12)
            else:
                _auto_근속년수 = 0
            st.caption("근속년수: " f"{_auto_근속년수}년")
        with b3:
            _auto_연금수령연차 = max(0, 연금개시일.year - _연금수령가능일_dt.year) + 6 if 퇴직연금제도가입일 < _date(2013, 1, 1) else 1            
            st.caption("연금개시일 연금수령연차: " f"{_auto_연금수령연차}")
        
        st.subheader("연금소득 재원(원)")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            과세제외_자기부담금 = st.number_input("과세제외 자기부담금", value=_def_과세제외_자기부담금, step=100_000)
        with a2:
            이연퇴직소득 = st.number_input("이연퇴직소득(= IRP 입금 퇴직금)", value=_def_이연퇴직소득, step=1_000_000)
        with a3:
            세액공제자기부담금 = st.number_input("세액공제자기부담금", value=_def_세액공제자기부담금, step=100_000)
        with a4:
            운용손익 = st.number_input("운용손익", value=_def_운용손익, step=100_000)

        st.caption(f"총평가금액(= 과세제외 자기부담금 + 이연퇴직소득 + 그외(=세액공제자기부담금 + 운용손익)): "
                   f"{과세제외_자기부담금 + 이연퇴직소득 + 세액공제자기부담금 + 운용손익:,} 원")
        calc_퇴직소득세 = pnsn_sim_calculator.calc_퇴직소득세(
            근속년수=_auto_근속년수, 이연퇴직소득=이연퇴직소득, 
        )
        st.caption(f"예상 퇴직소득세율(이연퇴직소득 기준): {calc_퇴직소득세['퇴직소득세율']:.1%}")
        st.caption(f"예상 퇴직소득세(이연퇴직소득 기준): {calc_퇴직소득세['퇴직소득산출세액']:,} 원")
        submitted_main = st.form_submit_button("기본 정보 저장")

    
    st.subheader("지급 옵션")
    c1, c2, c3 = st.columns(3)    
    with c1:
        지급옵션 = st.selectbox("지급옵션", ["기간확정형", "금액확정형", "한도수령", "최소수령", "일시금"],
                                index=0, key="지급옵션")

    if 지급옵션 == "기간확정형":
        with c2:
            지급기간_년 = st.number_input(
                "지급기간_년(필수)", 
                min_value=1, 
                value=st.session_state.get("지급기간_년", 10), 
                step=1 
            )
        수령금액_년 = None

    elif 지급옵션 == "금액확정형":
        with c2:
            수령금액_년 = st.number_input(
                "수령금액_년(필수, 원)", 
                min_value=1, 
                value=12_000_000, 
                step=100_000
            )
        지급기간_년 = None

    else:
        # 한도수령, 최소수령일 경우
        지급기간_년, 수령금액_년 = None, None

    submitted_option = st.button("시뮬레이션 실행")

    if submitted_option:
        params = dict(
            평가기준일=평가기준일,
            # ↓ pnsn_sim_calculator.simulate_pension이 '연금수령가능일'을 직접 쓰는 구조라면 이 값을 사용
            연금개시일=연금개시일,
            # (만약 내부에서 C25/C26로 재계산한다면 생년월일/퇴직일/IRP가입일을 넘기고 이 키는 빼세요)
            생년월일=생년월일,
            입사일=입사일,
            퇴직일=퇴직일,
            퇴직연금제도가입일=퇴직연금제도가입일,
            IRP가입일=IRP가입일,

            운용수익률=float(운용수익률),
            과세제외_자기부담금=int(과세제외_자기부담금),
            이연퇴직소득=int(이연퇴직소득),
            그외=int(세액공제자기부담금 + 운용손익),

            지급옵션=지급옵션,
            지급기간_년=int(지급기간_년) if 지급기간_년 else None,
            수령금액_년=int(수령금액_년) if 수령금액_년 else None,
        )

        # 필수 검증
        if 지급옵션 == "기간확정형" and not params["지급기간_년"]:
            st.error("기간확정형에는 '지급기간_년'이 필요합니다."); st.stop()
        if 지급옵션 == "금액확정형" and not params["수령금액_년"]:
            st.error("금액확정형에는 '수령금액_년'이 필요합니다."); st.stop()

        try:
            with st.spinner("계산 중..."):
                df_capped = pnsn_sim_calculator.simulate_pension(**params)
                
            # 입력값 요약 + 결과 출력
            st.subheader("산출결과")
            # nice = params.copy()
            # # 날짜를 문자열로 바꿔 보기 좋게
            # for k in ["평가기준일","연금수령가능일","생년월일","입사일","퇴직일","퇴직연금제도가입일","IRP가입일"]:
            #     if isinstance(nice.get(k), _date):
            #         nice[k] = str(nice[k])
            # st.dataframe(style_dataframe(pd.DataFrame([nice])), hide_index=True)
            

            m1, m2, m3, m4 = st.columns(4)
            _auto_현재나이 = (평가기준일.year - 생년월일.year) - \
                             (1 if (연금개시일.month, 연금개시일.day) < (생년월일.month, 생년월일.day) else 0)            
            with m1: st.metric("현재연령", f"{_auto_현재나이} 세")                    
            with m2: st.metric("연금개시일자", f"{연금개시일}")
            with m3: st.metric("연금개시연령", f"{_auto_수령나이}세")
            with m4: st.metric("연금개시금액", f"{int(df_capped[df_capped['지급회차']==1]['지급전잔액'].values[0]):,} 원")

            if {"총세액","실수령액","실제지급액"}.issubset(df_capped.columns):
                m1, m2, m3, m4 = st.columns(4)
                with m1: st.metric("총 연금수령액", f"{int(df_capped['실제지급액'].sum()):,} 원")                    
                with m2: st.metric("총 세액 합계", f"{int(df_capped['총세액'].sum()):,} 원")
                with m3: st.metric("실수령 합계", f"{int(df_capped['실수령액'].sum()):,} 원")
                eff_tax_rate = df_capped['총세액'].sum() / df_capped['실제지급액'].sum() if df_capped['실제지급액'].sum() > 0 else 0
                with m4: st.metric("실효세율", f"{eff_tax_rate:.1%}")

            st.markdown("##### 산출결과 내역")
            # col_view = ["지급회차","나이","지급전잔액","한도","실제지급액","총세액","실수령액","세율","지급옵션"]
            # st.dataframe(
            #     style_dataframe(df_capped[col_view]),
            #     use_container_width=True, 
            #     hide_index=True,
            #     )
            # 1) 컬럼 생성
            df_capped["한도초과여부"] = df_capped.apply(
                lambda x: (
                    "한도 이내" if pd.isna(x["한도"]) or x["한도"] >= x["실제지급액"] 
                    else "한도 초과"
                ),
                axis=1
            )

            # 2) 스타일 적용 (DataFrame 먼저 자른 후 .style 사용)
            col_view = ["지급회차","나이","지급전잔액","한도","실제지급액",
                        "총세액","실수령액","세율","지급옵션","한도초과여부"]

            styled_df = style_dataframe(df_capped[col_view]).map(
                lambda v: "color:green;" if v=="한도 이내" else "color:red;",
                subset=["한도초과여부"]
            )

            # 3) 출력
            st.dataframe(styled_df, use_container_width=True, hide_index=True)       

            st.markdown("##### 산출결과 세부내역")
            st.dataframe(
                style_dataframe(df_capped),
                column_config={
                "연금지급일": st.column_config.DateColumn("연금지급일", format="YYYY-MM-DD"),
                "과세기간개시일": st.column_config.DateColumn("과세기간개시일", format="YYYY-MM-DD"),
                },
                use_container_width=True, 
                hide_index=True,
                )

            st.download_button(
                "CSV 다운로드",
                data=df_capped.to_csv(index=False).encode("utf-8-sig"),
                file_name="연금시뮬레이션_df_capped.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error("시뮬레이션 중 오류가 발생했습니다.")
            st.exception(e)



# ----(3) 사이드바: 채팅 표시 + 입력 폼----
with st.sidebar:
    st.markdown("---")
    st.caption("대화 내역")

    # 대화 내역 표시 (사이드바 컨테이너)
    chat_box = st.container()
    with chat_box:
        for m in st.session_state.messages:
            if m["role"] == "user":
                st.markdown(f"**👤 You:**\n\n{m['content']}") 
            else:
                st.markdown(f"**🤖 Assistant:**\n\n{m['content']}")

    st.markdown("---")
    # 입력 폼: Enter로 전송 가능
    with st.form(key="chat_form", clear_on_submit=True):
        prompt = st.text_area(
            "메시지 입력", height=80, placeholder="여기에 입력하고 Enter 또는 Send"
        )
        send_btn = st.form_submit_button("Send")

    # ----(4) 전송 로직----
    if send_btn and prompt.strip():
        # 1) 사용자 메시지 저장/표시
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2) OpenAI 호출 (비-스트리밍: 사이드바에서도 안정적)
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": "You are a helpful, concise assistant.",
                    },
                    *st.session_state.messages,
                ],
            )
            answer = resp.output_text
        except Exception as e:
            answer = f"(오류) {type(e).__name__}: {e}"

        # 3) 어시스턴트 메시지 저장
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # 4) 즉시 리렌더링하여 사이드바 대화 갱신
        st.rerun()

# %%





