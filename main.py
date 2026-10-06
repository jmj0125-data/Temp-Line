import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# ==================================================
# 기본 설정
# ==================================================

st.set_page_config(
    page_title="기온 예측기 - 곡선 비교",
    page_icon="🌡️",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199b10d3e7670e30231bce4/data/seoul.csv"
)

LAST_YEAR = 2025
MIN_OBSERVATION_DAYS = 300

TRAIN_END = 2004
TEST_START = 2005

BASE_YEAR = 1908


# ==================================================
# 데이터 불러오기
# ==================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

    # 날짜
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["날짜", "평균기온"]
    )

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 2025년까지만 사용
    df = df[
        df["연도"] <= LAST_YEAR
    ]

    # 연도별 평균기온 + 관측일수
    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일 300일 미만인 해 제외
    annual = annual[
        annual["관측일수"] >= MIN_OBSERVATION_DAYS
    ].copy()

    annual = annual.sort_values("연도")

    return annual.reset_index(drop=True)


annual = load_data()


# ==================================================
# 훈련 / 테스트 데이터 분리
# ==================================================

train = annual[
    annual["연도"] <= TRAIN_END
].copy()

test = annual[
    annual["연도"] >= TEST_START
].copy()


# ==================================================
# 고차식 계산을 위한 연도 변환
# ==================================================
#
# 실제 연도(1900, 2000 등)를 그대로 9차식에 넣으면
# x^9의 숫자가 지나치게 커질 수 있다.
#
# 따라서 "기준 연도에서 몇 년 지났는가"로 바꾼다.
#
# 예:
# 1908년 → 0
# 1909년 → 1
# 2005년 → 97
# 2050년 → 142
#
# 그리고 PolynomialFeatures가 만드는 다항식에
# StandardScaler를 사용하지 않고도 안정적으로 계산할 수
# 있도록 x의 크기를 100년 기준으로 한 번 더 축소한다.
# ==================================================

scale = 100.0

train_x = (
    (train["연도"] - BASE_YEAR) / scale
).to_numpy().reshape(-1, 1)

train_y = train["연평균기온"].to_numpy()

test_x = (
    (test["연도"] - BASE_YEAR) / scale
).to_numpy().reshape(-1, 1)

test_y = test["연평균기온"].to_numpy()


# ==================================================
# 다항회귀 모델 만들기
# ==================================================

def make_polynomial_model(degree):

    return make_pipeline(
        PolynomialFeatures(
            degree=degree,
            include_bias=False
        ),
        LinearRegression()
    )


models = {}

predictions = {}

results = []


# ==================================================
# 1차 / 3차 / 9차 모델 학습 및 테스트
# ==================================================

for degree in [1, 3, 9]:

    model = make_polynomial_model(degree)

    # 반드시 훈련 데이터만으로 학습
    model.fit(
        train_x,
        train_y
    )

    # 테스트 데이터 예측
    test_pred = model.predict(
        test_x
    )

    # 테스트 데이터 MAE
    mae = mean_absolute_error(
        test_y,
        test_pred
    )

    # 2050년 예측
    x_2050 = np.array(
        [[
            (2050 - BASE_YEAR) / scale
        ]]
    )

    prediction_2050 = float(
        model.predict(x_2050)[0]
    )

    models[degree] = model
    predictions[degree] = test_pred

    results.append(
        {
            "모델": f"{degree}차",
            "테스트 평균 오차 (MAE, ℃)": mae,
            "2050년 예상 연평균기온 (℃)": prediction_2050
        }
    )


results_df = pd.DataFrame(results)


# ==================================================
# 화면
# ==================================================

st.title("🌡️ 기온 예측기 — 직선과 곡선 비교")

st.write(
    "과거 서울의 연평균기온을 이용해 1차, 3차, 9차 "
    "다항회귀 모델을 학습하고, 학습에 사용하지 않은 "
    "테스트 데이터로 예측 성능을 비교합니다."
)


# ==================================================
# 데이터 분할 표시
# ==================================================

st.subheader("📚 훈련 데이터와 테스트 데이터")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 🧑‍🏫 훈련 데이터")

    st.metric(
        "연도 수",
        f"{len(train)}개"
    )

    st.write(
        f"사용 기간: "
        f"**{int(train['연도'].min())}~"
        f"{int(train['연도'].max())}년**"
    )

with col2:

    st.markdown("### 📝 테스트 데이터")

    st.metric(
        "연도 수",
        f"{len(test)}개"
    )

    st.write(
        f"사용 기간: "
        f"**{int(test['연도'].min())}~"
        f"{int(test['연도'].max())}년**"
    )

st.info(
    "중요: 테스트 데이터는 1차·3차·9차 모델의 학습에 "
    "전혀 사용하지 않았습니다. 모델을 학습한 뒤 "
    "테스트 데이터에 처음 적용하여 성능을 평가합니다."
)


# ==================================================
# 핵심 결과 표
# ==================================================

st.subheader("📊 모델별 예측 결과")

display_df = results_df.copy()

display_df[
    "테스트 평균 오차 (MAE, ℃)"
] = display_df[
    "테스트 평균 오차 (MAE, ℃)"
].map(
    lambda x: f"{x:.3f}"
)

display_df[
    "2050년 예상 연평균기온 (℃)"
] = display_df[
    "2050년 예상 연평균기온 (℃)"
].map(
    lambda x: f"{x:.2f}"
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "MAE는 테스트 기간 동안 실제 기온과 예측 기온의 "
    "평균적인 차이입니다. 작을수록 예측이 좋습니다."
)


# ==================================================
# 가장 좋은 모델 표시
# ==================================================

best_row = results_df.loc[
    results_df[
        "테스트 평균 오차 (MAE, ℃)"
    ].idxmin()
]

best_model_name = best_row["모델"]
best_mae = best_row[
    "테스트 평균 오차 (MAE, ℃)"
]

st.success(
    f"테스트 데이터에서 평균 오차가 가장 작은 모델은 "
    f"**{best_model_name}**이며, "
    f"평균적으로 약 **{best_mae:.2f}℃** 차이가 났습니다."
)


# ==================================================
# 테스트 데이터에서 실제값과 예측값 비교
# ==================================================

st.subheader(
    "📈 테스트 기간 실제 기온 vs 모델 예측"
)

fig = go.Figure()


# 실제 기온
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 연평균기온",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "실제 기온: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)


# 1차
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=predictions[1],
        mode="lines",
        name="1차 모델",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "1차 예측: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)


# 3차
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=predictions[3],
        mode="lines",
        name="3차 모델",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "3차 예측: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)


# 9차
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=predictions[9],
        mode="lines",
        name="9차 모델",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "9차 예측: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified",
    legend_title="데이터"
)

fig.update_xaxes(
    tickformat="d"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==================================================
# 전체 기간에 대한 학습 곡선 시각화
# ==================================================

st.subheader(
    "📉 훈련 데이터에서 만든 세 가지 회귀곡선"
)

plot_years = np.arange(
    int(annual["연도"].min()),
    2051
)

plot_x = (
    (plot_years - BASE_YEAR) / scale
).reshape(-1, 1)

fig2 = go.Figure()


# 실제 데이터
fig2.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=6
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "평균기온: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)


# 훈련 / 테스트 경계 표시
fig2.add_vline(
    x=TEST_START,
    line_dash="dash",
    annotation_text="훈련 → 테스트",
    annotation_position="top"
)


for degree in [1, 3, 9]:

    curve = models[degree].predict(
        plot_x
    )

    fig2.add_trace(
        go.Scatter(
            x=plot_years,
            y=curve,
            mode="lines",
            name=f"{degree}차 회귀곡선",
            hovertemplate=(
                "<b>%{x}년</b><br>"
                f"{degree}차 예측: "
                "%{y:.2f} ℃"
                "<extra></extra>"
            )
        )
    )


fig2.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    legend_title="모델"
)

fig2.update_xaxes(
    tickformat="d"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)


# ==================================================
# 2050년 예측 비교
# ==================================================

st.subheader("🔮 2050년 예측 비교")

c1, c2, c3 = st.columns(3)

for col, degree in zip(
    [c1, c2, c3],
    [1, 3, 9]
):

    prediction = results_df.loc[
        results_df["모델"] == f"{degree}차",
        "2050년 예상 연평균기온 (℃)"
    ].iloc[0]

    with col:

        st.markdown(
            f"### {degree}차 모델"
        )

        st.metric(
            "2050년 예상 기온",
            f"{prediction:.2f} ℃"
        )


# ==================================================
# 해석
# ==================================================

st.subheader("💡 어떻게 비교하면 될까?")

st.markdown(
    """
- **MAE가 작은 모델**: 테스트 기간의 실제 기온을 평균적으로 더 가깝게 예측한 모델입니다.
- **1차 모델**: 변화 추세를 하나의 직선으로 표현합니다.
- **3차 모델**: 기온 변화가 한 번 휘어지는 정도까지 표현할 수 있습니다.
- **9차 모델**: 과거 데이터의 복잡한 굴곡을 훨씬 자유롭게 따라갈 수 있습니다.
- 하지만 **차수가 높다고 반드시 미래 예측을 더 잘하는 것은 아닙니다.**
  특히 9차 모델은 훈련 데이터에 지나치게 맞춰질 가능성이 있기 때문에
  이번처럼 **학습에 사용하지 않은 테스트 데이터의 MAE로 평가하는 것**이 중요합니다.
"""
)

st.caption(
    "9차 회귀에서는 연도 자체를 그대로 사용하지 않고 "
    "1908년을 기준으로 한 경과 연수를 100으로 나눈 값을 사용해 "
    "고차식 계산에서 발생할 수 있는 수치적인 불안정을 줄였습니다."
)
