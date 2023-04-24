import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import librosa
import math

st.title('BearingMind')
st.subheader('Developed by. 김영진')

df = pd.read_csv(r'bearings_.csv')

#옵션 선택박스
model_list = list(set((df["pkmt"]+'\t'+df["model"]).tolist()))  # model과 pkmt를 합쳐서 option_list 생성
model_list.sort()  # 옵션 정렬
option_1 = st.sidebar.selectbox('Choose Model', [''] + model_list)
if option_1:
    spec_list = list(set(df.loc[df['pkmt']==option_1[:6],'spd_spec']))
    option_2 = st.sidebar.selectbox('Choose Spindle Spec',[''] + spec_list)
    option_3 = st.sidebar.text_input('Input Spindle RPM')   


# 파일 업로드 위젯
uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

# 파일이 업로드 되었을 경우에만 실행
if st.button("Diagnosis") and uploaded_file and option_3 is not None:
    #제원 정보 추출
    brg_data = df[(df["pkmt"] == option_1[:6])&(df['spd_spec']==option_2)]
    if not brg_data.empty:
        ##결함주파수 계산
        def bpfo_func(row, spd_rpm):
            return (spd_rpm/60)*row['num_of_ball']/2*(1-row['dia_of_ball']/row['pcd']*math.cos(math.pi*row['contact_angle']/180))
        def bpfi_func(row, spd_rpm):
            return (spd_rpm/60)*row['num_of_ball']/2*(1+row['dia_of_ball']/row['pcd']*math.cos(math.pi*row['contact_angle']/180))
        def fdf_func(row, spd_rpm):
            return (spd_rpm/60)/2*(1-row['dia_of_ball']/row['pcd']*math.cos(math.pi*row['contact_angle']/180))
        def bsf_func(row, spd_rpm):
            return (spd_rpm/60)*row['pcd']/(2*row['dia_of_ball'])*(1-((row['dia_of_ball']/row['pcd'])**2)*(math.cos(math.pi*row['contact_angle']/180)**2))
        
        spd_rpm = int(option_3)
        brg_data['bpfo'] = brg_data.apply(bpfo_func, args=(spd_rpm,), axis=1)
        brg_data['bpfi'] = brg_data.apply(bpfi_func, args=(spd_rpm,), axis=1)
        brg_data['fdf'] = brg_data.apply(fdf_func, args=(spd_rpm,), axis=1)
        brg_data['bsf'] = brg_data.apply(bsf_func, args=(spd_rpm,), axis=1)
        # st.write(brg_data[['bpfo','bpfi','fdf','bsf']])

    else:
        st.write("선택하신 옵션의 베어링 제원정보가 없으므로 베어링 결함 진단은 생략합니다.")
        st.write("선택사양 : {}".format(option_1[:6]))

    # WAV 파일에서 sampling rate와 data 추출
    y, sr = librosa.load(uploaded_file)

    # 재생시간을 기준으로 중간 2초 분량 추출
    center = len(y) // 2
    length = sr * 2
    start = center - (length // 2)
    end = center + (length // 2)
    y_cut = y[start:end]

    # 퓨리에 변환 적용
    N = len(y_cut)
    yf = np.fft.fft(y_cut)
    xf = np.linspace(0.0, sr/2, N//2)

    # 절댓값으로 변환하여 그래프 데이터 생성
    yf_abs = np.abs(yf[:N//2])

    # 그래프 출력
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xf, y=yf_abs))
    fig.update_xaxes(title_text='Frequency (Hz)')
    fig.update_yaxes(title_text='Magnitude')
    fig.update_layout(title='Frequency domain waveform')
    fig.update_xaxes(range=[0, sr/2])
    fig.update_yaxes(range=[0, max(yf_abs)*1.1])
    st.plotly_chart(fig)
