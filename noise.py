import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import librosa

st.title('Spindle Bearing Diagnosis System')
st.subheader('Developed by. 김영진')
df = pd.read_csv(r'bearings_.csv')
model_list = list(set((df["pkmt"]+'\t'+df["model"]).tolist()))  # model과 pkmt를 합쳐서 option_list 생성
model_list.sort()  # 옵션 정렬
option_1 = st.sidebar.selectbox('Choose Model', [''] + model_list)

if option_1:
    spec_list = list(set(df.loc[df['pkmt']==option_1[:6],'spd_spec']))
    option_2 = st.sidebar.selectbox('Choose Spindle Spec',[''] + spec_list)

# 파일 업로드 위젯 생성
uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

# 파일이 업로드 되었을 경우에만 실행
if st.button("Diagnosis") and uploaded_file is not None:
    # WAV 파일에서 sampling rate와 data 추출
    wav, sr = librosa.load(uploaded_file)

    #FFT 변환
    fft = np.fft.fft(wav) 
    magnitude = np.abs(fft)
    frequency = np.linspace(0, sr, len(magnitude))
    left_frequency = frequency[:int(len(frequency)/2)]
    left_magnitude = magnitude[:int(len(magnitude)/2)]/1000
    freq_range = int(len(left_magnitude))

    # 파형 그래프 생성 (주파수 도메인에서의 파형)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=left_frequency, y=left_magnitude))
    fig.update_xaxes(title_text='Frequency (Hz)')
    fig.update_yaxes(title_text='Magnitude')
    fig.update_layout(title='Frequency domain waveform')
    st.plotly_chart(fig) # 그래프 출력

    #제원 정보 출력
    brg_data = df[df["model"] == option_1]
    if not brg_data.empty:
        ball_dia = brg_data["dia_of_ball"].values[0]
        ball_n = brg_data["num_of_ball"].values[0]
        angle = brg_data["contact_angle"].values[0]
        pcd = brg_data["pcd"].values[0]

        text = "{option_1}의 볼 직경은 {ball_dia}이고, 볼 갯수는 {ball_n}이며, 접촉각은 {angle}이고 pcd는 {pcd}입니다."
        st.write(text)

    else:
        st.write("")
