import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# VitoRadar - AI Powered Contactless Vital Signs Demo
# ============================================================

st.set_page_config(
    page_title="VitoRadar AI Dashboard",
    page_icon="📡",
    layout="wide"
)

# ------------------------------------------------------------
# Signal generation
# ------------------------------------------------------------

def generate_normal(duration=30, sampling_rate=50):
    t = np.arange(0, duration, 1 / sampling_rate)
    # Normal breathing: approximately 15 breaths/min
    breathing = np.sin(2 * np.pi * 0.25 * t)
    noise = 0.08 * np.random.randn(len(t))
    return t, breathing + noise


def generate_apnea(duration=30, sampling_rate=50):
    t = np.arange(0, duration, 1 / sampling_rate)
    signal = np.sin(2 * np.pi * 0.25 * t)
    # Simulate apnea in middle section
    start = int(len(t) * 0.35)
    end = int(len(t) * 0.75)
    signal[start:end] = 0
    noise = 0.03 * np.random.randn(len(t))
    return t, signal + noise


def generate_irregular(duration=30, sampling_rate=50):
    t = np.arange(0, duration, 1 / sampling_rate)
    signal = (
        0.7 * np.sin(2 * np.pi * 0.15 * t)
        + 0.5 * np.sin(2 * np.pi * 0.40 * t)
    )
    noise = 0.15 * np.random.randn(len(t))
    return t, signal + noise


def generate_motion(duration=30, sampling_rate=50):
    t = np.arange(0, duration, 1 / sampling_rate)
    breathing = np.sin(2 * np.pi * 0.25 * t)
    motion = (
        2 * np.sin(2 * np.pi * 0.05 * t)
        + 1.5 * np.random.randn(len(t))
    )
    return t, breathing + motion


# ------------------------------------------------------------
# Signal filtering (band-pass 0.05 - 2 Hz, SOS form = stable)
# ------------------------------------------------------------

def filter_signal(signal, sampling_rate=50):
    low = 0.05
    high = 2.0
    nyquist = sampling_rate / 2
    sos = butter(
        3,
        [low / nyquist, high / nyquist],
        btype="band",
        output="sos"
    )
    return sosfiltfilt(sos, signal)


# ------------------------------------------------------------
# FFT
# ------------------------------------------------------------

def calculate_fft(signal, sampling_rate=50):
    fft_result = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(len(signal), 1 / sampling_rate)
    magnitude = np.abs(fft_result)
    return frequencies, magnitude


# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------

def extract_features(signal, sampling_rate=50):
    frequencies, magnitude = calculate_fft(signal, sampling_rate)
    magnitude[0] = 0
    peak_index = np.argmax(magnitude)
    dominant_frequency = frequencies[peak_index]
    breathing_rate = dominant_frequency * 60
    amplitude = np.max(signal) - np.min(signal)
    variance = np.var(signal)
    rms = np.sqrt(np.mean(signal ** 2))
    return [dominant_frequency, breathing_rate, amplitude, variance, rms]


# ------------------------------------------------------------
# Create demo AI model
# ------------------------------------------------------------

@st.cache_resource
def create_model():
    X = []
    y = []

    generators = [
        (generate_normal, "NORMAL"),
        (generate_apnea, "APNEA"),
        (generate_irregular, "IRREGULAR"),
        (generate_motion, "MOTION_ARTIFACT"),
    ]

    for generator, label in generators:
        for _ in range(100):
            _, signal = generator()
            signal = filter_signal(signal)
            X.append(extract_features(signal))
            y.append(label)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


model = create_model()


# ============================================================
# DASHBOARD
# ============================================================

st.title("📡 VitoRadar AI-Powered Vital Signs Monitoring")

st.markdown(
    """
    **Contactless Vital Signs Monitoring using Ku-Band Sensing**

    This software prototype demonstrates the signal-processing
    and AI stages using simulated radar-like data.
    """
)

st.divider()

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.header("Patient Monitoring")

patient_name = st.sidebar.text_input("Patient Name", "Demo Patient")

scenario = st.sidebar.selectbox(
    "Select Patient Condition",
    ["Normal", "Apnea", "Irregular", "Motion Artifact"]
)

run_demo = st.sidebar.button("▶ Run Demo", type="primary")


# ============================================================
# DEMO
# ============================================================

if run_demo:

    # Generate signal
    if scenario == "Normal":
        t, raw_signal = generate_normal()
    elif scenario == "Apnea":
        t, raw_signal = generate_apnea()
    elif scenario == "Irregular":
        t, raw_signal = generate_irregular()
    else:
        t, raw_signal = generate_motion()

    # Filter signal
    filtered_signal = filter_signal(raw_signal)

    # Extract features
    features = extract_features(filtered_signal)
    dominant_frequency = features[0]
    breathing_rate = features[1]
    amplitude = features[2]
    variance = features[3]
    rms = features[4]

    # AI prediction
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    confidence = np.max(probabilities) * 100

    # Dashboard metrics
    st.subheader(f"Patient: {patient_name}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Breathing Rate", f"{breathing_rate:.1f} BPM")
    with col2:
        st.metric("Dominant Frequency", f"{dominant_frequency:.2f} Hz")
    with col3:
        st.metric("Signal Variance", f"{variance:.3f}")
    with col4:
        st.metric("AI Confidence", f"{confidence:.1f}%")

    st.divider()

    # Signal graph
    st.subheader("📈 Radar Chest-Wall Motion Signal")

    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(t, raw_signal, label="Raw Radar Signal", alpha=0.5)
    ax1.plot(t, filtered_signal, label="Filtered Signal")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Signal Amplitude")
    ax1.set_title("Simulated Radar Chest-Wall Motion")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # FFT graph
    st.subheader("📊 FFT Frequency Spectrum")

    frequencies, magnitude = calculate_fft(filtered_signal)
    mask = frequencies <= 2  # breathing range only

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(frequencies[mask], magnitude[mask])
    ax2.axvline(
        dominant_frequency,
        linestyle="--",
        label=f"Peak = {dominant_frequency:.2f} Hz"
    )
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("FFT Magnitude")
    ax2.set_title("Fast Fourier Transform - Breathing Frequency")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # AI result
    st.divider()
    st.subheader("🤖 AI Classification Result")

    if prediction == "NORMAL":
        st.success(
            "✅ NORMAL\n\n"
            "Patient breathing pattern appears normal."
        )
    elif prediction == "APNEA":
        st.error(
            "🚨 APNEA DETECTED\n\n"
            "Possible breathing interruption detected."
        )
    elif prediction == "IRREGULAR":
        st.warning(
            "⚠️ IRREGULAR BREATHING\n\n"
            "Abnormal breathing pattern detected."
        )
    else:
        st.warning(
            "⚠️ MOTION ARTIFACT\n\n"
            "Body movement may affect the measurement."
        )

    # Feature table
    st.subheader("🔍 Extracted Signal Features")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:
        st.write(f"**Dominant Frequency:** {dominant_frequency:.3f} Hz")
        st.write(f"**Breathing Rate:** {breathing_rate:.2f} BPM")
        st.write(f"**Signal Amplitude:** {amplitude:.3f}")

    with feature_col2:
        st.write(f"**Variance:** {variance:.4f}")
        st.write(f"**RMS:** {rms:.4f}")
        st.write(f"**AI Prediction:** {prediction}")

else:

    # Initial dashboard
    st.info(
        "👈 Select a condition from the left side and "
        "click **Run Demo**."
    )

    st.subheader("VitoRadar Demo Architecture")

    st.markdown(
        """
        **Antenna → RF Detector → Filter → ESP32 → FFT → AI Model → Dashboard**

        Current prototype:

        **Simulated Radar Signal → Signal Processing → FFT → Random Forest → Dashboard**
        """
    )

    st.warning(
        "Note: This demonstration uses simulated radar-like "
        "signals. The physical Ku-band antenna is the hardware "
        "component; the RF detector, filtering and ESP32 "
        "integration are planned for the next stage."
    )
