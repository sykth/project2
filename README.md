# 🌡️ 도시 열섬 3D 돔 시뮬레이터

반구 돔을 1 m 정육면체 격자로 채우고, 왼쪽은 **숲**, 오른쪽은 **아스팔트 + 아파트**로 나눠서
하루(24시간) 동안의 온도 변화를 계산하는 모형이야. 일반 옥상 / 쿨루프 / 옥상 녹화를 메뉴에서 바꿔 비교할 수 있어.

## 폴더 구조
```
uhi-streamlit/
├─ app.py            # Streamlit 웹앱 (메뉴, 3D 그림, 그래프)
├─ uhi_core.py       # 열수지 시뮬레이션 핵심 (순수 numpy)
├─ geometry.py       # 정육면체 격자 / 아파트 직육면체 면 만들기
├─ viz.py            # plotly 그림 함수
├─ precompute.py     # 시뮬레이션 결과를 data/results.npz 로 저장
├─ data/results.npz  # 미리 계산해 둔 결과 (웹에서 바로 불러옴)
├─ requirements.txt
└─ .streamlit/config.toml
```

## 1. 내 컴퓨터에서 먼저 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```
브라우저가 자동으로 열려. (배포 전에 꼭 한 번 로컬에서 확인해 보자.)

## 2. GitHub 에 올리기
```bash
cd uhi-streamlit
git init
git add .
git commit -m "도시 열섬 3D 돔 시뮬레이터"
git branch -M main
git remote add origin https://github.com/<내-아이디>/<저장소-이름>.git
git push -u origin main
```
GitHub 에서 먼저 빈 저장소(Repository)를 만들어 두면 돼. `data/results.npz` 는 약 0.7 MB 라 그냥 올려도 괜찮아.

## 3. Streamlit Community Cloud 로 배포 (무료)
1. https://share.streamlit.io 에 GitHub 계정으로 로그인
2. **Create app → Deploy a public app from GitHub**
3. Repository: `<내-아이디>/<저장소-이름>`, Branch: `main`, Main file path: `app.py`
4. **Deploy** → 몇 분 뒤 `https://....streamlit.app` 주소가 생겨

## 4. 설정값을 바꿔서 다시 계산하려면
`uhi_core.py` 위쪽 설정값(알베도 `ALB`, 증발산 `ET`, 건물 크기 `BLD_SIZE`, 환기 `TAU_BG` 등)을 고친 다음
```bash
python precompute.py
```
를 실행하면 `data/results.npz` 가 새로 만들어져. 그 파일을 다시 커밋해서 push 하면 웹에도 반영돼.
(결과 파일이 없으면 웹앱이 직접 계산하지만 1~3분 걸려서, 항상 미리 계산해서 올리는 걸 추천해.)

## 모형 요약 (자세한 설명과 한계는 앱의 '모형 설명 · 한계' 탭)
- 격자마다 열수지: 태양 흡수 − 증발산 − 복사냉각 + 전도/대류 + 인공열
- 아파트는 격자가 아닌 직육면체 한 덩어리(벽 + 옥상 노드), 주변 공기 격자와만 열 교환
- 바람은 열섬 순환(온도 차이로 생김) + 숲 → 도시 배경 바람이고, 열 이동 계산에도 쓰임
- 일부 계수(`SOLID_ACTIVE`, `K_AIR`, `TAU_BG` …)는 현실적인 열섬 크기에 맞춘 **보정값**이라, 절대 온도 예측용이 아니라 재질 차이의 영향을 보여 주는 정성적 모형이야.
