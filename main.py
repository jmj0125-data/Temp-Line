
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

BASE_YEAR = 1908
LAST_CLASS_YEAR = 2025
MIN_OBSERVATION_DAYS = 300


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 2025년까지만 사용
    df = df[df["연도"] <= LAST_CLASS_YEAR]

    # 연도별 평균기온과 관측일수
    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )

    # 관측일 300일 미만인 해 제외
    annual = annual[
        annual["관측일수"] >= MIN_OBSERVATION_DAYS
    ].copy()

    annual = annual.dropna(subset=["연평균기온"])

    # 1908년부터 지난 연수
    annual["1908년부터_지난_연수"] = (
        annual["연도"] - BASE_YEAR
    )

    return annual


annual = load_data()


# --------------------------------------------------
# 전체 기간 회귀
# --------------------------------------------------
x_all = annual["1908년부터_지난_연수"].to_numpy()
y_all = annual["연평균기온"].to_numpy()

slope_all, intercept_all = np.polyfit(x_all, y_all, 1)

annual["전체기간_회귀기온"] = (
    slope_all * x_all + intercept_all
)

# 1년에 몇 도 상승 → 100년에 몇 도 상승
slope_all_100 = slope_all * 100


# --------------------------------------------------
# 최근 20년 회귀
# --------------------------------------------------
recent_20 = annual.tail(20).copy()

x_recent = (
    recent_20["연도"] - BASE_YEAR
).to_numpy()

y_recent = recent_20["연평균기온"].to_numpy()

slope_recent, intercept_recent = np.polyfit(
    x_recent,
    y_recent,
    1
)

slope_recent_100 = slope_recent * 100


# --------------------------------------------------
# 상관계수
# --------------------------------------------------
correlation = annual["연도"].corr(
    annual["연평균기온"]
)


# --------------------------------------------------
# 기간 정보
# --------------------------------------------------
start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())
year_count = len(annual)

recent_start_year = int(recent_20["연도"].min())
recent_end_year = int(recent_20["연도"].max())


# --------------------------------------------------
# 화면
# --------------------------------------------------
st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연도별 평균기온을 이용해 기온 변화 추세를 살펴보고, "
    "선형 회귀로 선택한 연도의 예상 평균기온을 계산합니다."
)


# --------------------------------------------------
# 전체 기간 / 최근 20년 기울기 비교
# --------------------------------------------------
st.subheader("📈 100년에 몇 ℃ 오르는가?")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 전체 기간")
    st.metric(
        f"{start_year}~{end_year}",
        f"{slope_all_100:+.2f} ℃ / 100년"
    )
    st.caption(
        f"{year_count}개 연도의 자료로 계산"
    )

with col2:
    st.markdown("### 최근 20년")
    st.metric(
        f"{recent_start_year}~{recent_end_year}",
        f"{slope_recent_100:+.2f} ℃ / 100년"
    )
    st.caption(
        "최근 20개 연도의 자료로 계산"
    )


# --------------------------------------------------
# 추가 분석 정보
# --------------------------------------------------
st.subheader("📊 분석에 사용한 기간")

info1, info2, info3, info4 = st.columns(4)

info1.metric(
    "전체 연도 수",
    f"{year_count}개"
)

info2.metric(
    "시작 연도",
    f"{start_year}년"
)

info3.metric(
    "끝 연도",
    f"{end_year}년"
)

info4.metric(
    "전체 상관계수",
    f"{correlation:.3f}"
)

st.caption(
    f"2025년 이후 데이터와 관측일이 "
    f"{MIN_OBSERVATION_DAYS}일 미만인 해는 제외했습니다."
)


# --------------------------------------------------
# 산점도 + 두 개의 회귀 직선
# --------------------------------------------------
st.subheader("📉 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균기온",
        customdata=annual["관측일수"],
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "평균기온: %{y:.2f} ℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        ),
    )
)


# 전체 기간 회귀선
line_years_all = np.arange(
    start_year,
    end_year + 1
)

line_x_all = line_years_all - BASE_YEAR

line_temperature_all = (
    slope_all * line_x_all
    + intercept_all
)

fig.add_trace(
    go.Scatter(
        x=line_years_all,
        y=line_temperature_all,
        mode="lines",
        name="전체 기간 회귀선",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "회귀선: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)


# 최근 20년 회귀선
line_years_recent = np.arange(
    recent_start_year,
    recent_end_year + 1
)

line_x_recent = line_years_recent - BASE_YEAR

line_temperature_recent = (
    slope_recent * line_x_recent
    + intercept_recent
)

fig.add_trace(
    go.Scatter(
        x=line_years_recent,
        y=line_temperature_recent,
        mode="lines",
        name="최근 20년 회귀선",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "최근 20년 회귀선: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    legend_title="데이터",
)

# 가로축에는 실제 연도 표시
fig.update_xaxes(
    tickformat="d",
    range=[
        start_year - 2,
        end_year + 2
    ],
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# 회귀식
# --------------------------------------------------
st.subheader("🧮 회귀 분석")

reg1, reg2 = st.columns(2)

with reg1:
    st.markdown("### 전체 기간")
    st.write(
        f"**100년당 변화량:** "
        f"{slope_all_100:+.2f} ℃"
    )
    st.write(
        f"**회귀식:** "
        f"예상 기온 = "
        f"{slope_all:.5f} × (연도 - {BASE_YEAR}) "
        f"{intercept_all:+.3f}"
    )

with reg2:
    st.markdown("### 최근 20년")
    st.write(
        f"**100년당 변화량:** "
        f"{slope_recent_100:+.2f} ℃"
    )
    st.write(
        f"**회귀식:** "
        f"예상 기온 = "
        f"{slope_recent:.5f} × (연도 - {BASE_YEAR}) "
        f"{intercept_recent:+.3f}"
    )


# --------------------------------------------------
# 연도 선택 및 예상 기온
# --------------------------------------------------
st.subheader("🔮 연도별 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

selected_x = selected_year - BASE_YEAR

predicted_temperature = (
    slope_all * selected_x
    + intercept_all
)

st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 30px;
        border-radius: 15px;
        background-color: rgba(128, 128, 128, 0.12);
        margin-top: 15px;
        margin-bottom: 20px;
    ">
        <div style="font-size: 24px;">
            {selected_year}년 예상 연평균기온
        </div>
        <div style="
            font-size: 60px;
            font-weight: bold;
            margin-top: 5px;
        ">
            {predicted_temperature:.2f} ℃
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if selected_year < start_year or selected_year > end_year:
    st.warning(
        f"{selected_year}년은 회귀 직선을 만든 관측 기간 "
        f"({start_year}~{end_year}년) 밖의 값입니다. "
        "따라서 회귀선을 연장한 추정값입니다."
    )

st.caption(
    "예상 기온은 과거 연평균기온의 선형 추세를 이용한 단순 추정값입니다."
)
