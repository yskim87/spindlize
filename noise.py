import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import librosa
import math
import joblib

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
if st.button("Diagnosis") and uploaded_file and option_3:
    # 제원 정보 추출
    brg_data = df[(df["pkmt"] == option_1[:6]) & (df['spd_spec'] == option_2)]
    if not brg_data.empty:
        # 결함주파수 계산 함수
        def calc_freqs(row, spd_rpm):
            bpfo = (spd_rpm / 60) * row['num_of_ball'] / 2 * (
                    1 - row['dia_of_ball'] / row['pcd'] * math.cos(math.pi * row['contact_angle'] / 180))
            bpfi = (spd_rpm / 60) * row['num_of_ball'] / 2 * (
                    1 + row['dia_of_ball'] / row['pcd'] * math.cos(math.pi * row['contact_angle'] / 180))
            fdf = (spd_rpm / 60) / 2 * (
                    1 - row['dia_of_ball'] / row['pcd'] * math.cos(math.pi * row['contact_angle'] / 180))
            bsf = (spd_rpm / 60) * row['pcd'] / (2 * row['dia_of_ball']) * (
                    1 - ((row['dia_of_ball'] / row['pcd']) ** 2) * (math.cos(math.pi * row['contact_angle'] / 180) ** 2))
            return pd.Series({'freq_type': ['bpfo', 'bpfi', 'fdf', 'bsf'], 'frequency': [bpfo, bpfi, fdf, bsf]})

        spd_rpm = int(option_3)
        freqs_df = brg_data.apply(calc_freqs, args=(spd_rpm,), axis=1)

        # 데이터프레임 합치기
        freqs_df = freqs_df.explode(['freq_type', 'frequency']).reset_index(drop=True)
        calc_df = brg_data.loc[brg_data.index.repeat(4)].reset_index(drop=True)
        combined_df = pd.concat([calc_df, freqs_df], axis=1)

        # 결과 출력
        # st.write(combined_df)
    else:
        st.write("선택하신 옵션의 베어링 제원정보가 없으므로 베어링 결함 진단은 생략합니다.")

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

    #RMS 추출
    rms = np.sqrt(np.mean(np.square(y_cut)))

    # 절댓값으로 변환하여 그래프 데이터 생성
    yf_abs = np.abs(yf[:N//2])
    
    # 조화(배수) 성분들의 peak 값 추출, 데이터 프레임에 추가
    freqs = combined_df['frequency'].values
    peaks_df = pd.DataFrame(columns=['mean_peak', 'max_peak', 'top5_mean_peak'])

    for freq in freqs:
        harmonic_freqs = [freq * i for i in range(1, 10)]
        freq_resol_ratio = 0.01 # 주파수 정확도 1% 설정, 이거때문에 오류날 수 있으니 참고
        peaks = []

        for f in harmonic_freqs:
            freq_range = (f - (freq_resol_ratio/2)*f, f + (freq_resol_ratio/2)*f) ## 주파수 범위 1%로 설정
            idx_range = np.where(np.logical_and(xf >= freq_range[0], xf <= freq_range[1]))
            if idx_range[0].size == 0:
                max_idx = None
                peak_val = 0
            else:
                max_idx = np.argmax(np.abs(yf[idx_range]))
                peak_freq = xf[idx_range][max_idx]
                peak_val = np.abs(yf[idx_range][max_idx])
            peaks.append([peak_freq, peak_val/rms])
        np_peaks = np.array(peaks)
        mean_peak = np.mean(np_peaks[:, 1], axis=0)
        max_peak = np.max(np_peaks[:, 1], axis=0)
        top5_mean_peak = np.mean(np.sort(np_peaks[:, 1])[-5:], axis=0)

        peaks_df.loc[len(peaks_df)] = [mean_peak, max_peak, top5_mean_peak]

    combined_df = pd.concat([combined_df, peaks_df], axis=1)

    # 모델 불러오기
    model = joblib.load('model.joblib')

    #스케일링
    scaler = joblib.load('scaler.pkl')
    combined_df[['mean_peak','max_peak','top5_mean_peak']] = scaler.transform(combined_df[['mean_peak','max_peak','top5_mean_peak']])

    #예측(판정)
    combined_df['result'] = model.predict(combined_df[['mean_peak','max_peak','top5_mean_peak']])

    # 결과 출력
    st.write(combined_df[['component', 'cmp_description', 'result']])

    # 그래프 출력
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xf, y=yf_abs))
    fig.update_xaxes(title_text='Frequency (Hz)')
    fig.update_yaxes(title_text='Magnitude')
    fig.update_layout(title='Frequency domain waveform')
    fig.update_xaxes(range=[0, sr/2])
    fig.update_yaxes(range=[0, max(yf_abs)*1.1])
    st.plotly_chart(fig)





