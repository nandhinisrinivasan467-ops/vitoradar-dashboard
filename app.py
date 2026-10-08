import io
import time
import wave
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# VitoRadar - Live Patient Monitoring Demo (simulated radar)
# One patient, one continuous timeline:
#   Normal -> Irregular -> Motion -> Normal -> Sudden Apnea
# ============================================================

st.set_page_config(page_title="VitoRadar Live Monitor", page_icon="📡", layout="wide")

FS = 20              # samples per second
STEP = 10            # samples per screen update (0.5 s)
WINDOW_SEC = 20      # seconds shown on the graph
APNEA_SEC = 6        # flat-signal time needed to raise apnea (demo value)

# (stage, seconds)
TIMELINE = [
    ("NORMAL", 20),
    ("IRREGULAR", 14),
    ("MOTION", 8),
    ("NORMAL", 16),
    ("APNEA", 16),
]

rng = np.random.default_rng()


# ------------------------------------------------------------
# Simulated radar signal (breathing + tiny heartbeat + noise)
# ------------------------------------------------------------

def radar_samples(stage, t):
    heart_hz = {"NORMAL": 1.2, "IRREGULAR": 1.5, "MOTION": 1.3, "APNEA": 1.15}[stage]
    heart = 0.12 * np.sin(2 * np.pi * heart_hz * t)

    if stage == "NORMAL":
        breath = np.sin(2 * np.pi * 0.25 * t)                       # 15 breaths/min
    elif stage == "IRREGULAR":
        breath = (0.6 * np.sin(2 * np.pi * 0.42 * t)                # fast + shallow
                  + 0.3 * np.sin(2 * np.pi * 0.13 * t))
    elif stage == "MOTION":
        breath = (np.sin(2 * np.pi * 0.25 * t)
                  + 2.5 * np.sin(2 * np.pi * 0.7 * t)               # body movement
                  + 1.5 * rng.standard_normal(len(t)))
    else:                                                           # APNEA
        breath = np.zeros_like(t)                                   # breathing stopped

    noise = 0.03 * rng.standard_normal(len(t))
    return breath + heart + noise


# ------------------------------------------------------------
# Signal analysis (moving average -> FFT)
# ------------------------------------------------------------

KERNEL = np.ones(FS) / FS


def smooth(x):
    # 1-second moving average: keeps breathing, removes the fast heartbeat
    return np.convolve(x, KERNEL, mode="valid")


def peak_freq(x, low, high):
    x = x - np.mean(x)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=4096))
    freqs = np.fft.rfftfreq(4096, 1 / FS)
    band = (freqs >= low) & (freqs <= high)
    return freqs[band][np.argmax(spectrum[band])]


def breathing_rate(valid):
    need = 12 * FS + FS - 1
    if len(valid) < need:
        return None
    sm = smooth(valid[-need:])
    return peak_freq(sm, 0.08, 0.8) * 60


def heart_rate(valid):
    need = 10 * FS + FS - 1
    if len(valid) < need:
        return None
    x = valid[-need:]
    sm = smooth(x)
    hp = x[FS // 2: FS // 2 + len(sm)] - sm        # remove breathing, keep heartbeat
    return peak_freq(hp, 0.8, 2.2) * 60


def is_apnea(valid):
    need = APNEA_SEC * FS + FS - 1
    if len(valid) < need:
        return False
    sm = smooth(valid[-need:])
    return (np.max(sm) - np.min(sm)) < 0.15


def is_motion(chunk):
    return np.max(np.abs(chunk)) > 1.6 or np.std(chunk) > 0.9


# ------------------------------------------------------------
# Alarm sound (siren WAV generated in code)
# ------------------------------------------------------------

def make_siren(seconds=4, sr=22050):
    t = np.arange(int(sr * seconds)) / sr
    freq = np.where((t * 2) % 1 < 0.5, 800, 1100)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    audio = (0.6 * np.sin(phase) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


# ------------------------------------------------------------
# Drawing
# ------------------------------------------------------------

LINE_COLOR = {
    "CALIBRATING": "#607d8b",
    "NORMAL": "#2e7d32",
    "IRREGULAR": "#ef6c00",
    "MOTION": "#607d8b",
    "APNEA": "#c62828",
}


def draw_chart(disp, status):
    win = np.array(disp[-WINDOW_SEC * FS:], dtype=float)
    x = np.linspace(-len(win) / FS, 0, len(win))

    fig, ax = plt.subplots(figsize=(10, 3.2))
    if status == "APNEA":
        ax.set_facecolor("#ffe5e5")

    ax.plot(x, win, color=LINE_COLOR[status], linewidth=1.4)

    blocked = np.isnan(win)
    if blocked.any():
        ax.fill_between(x, -1.8, 1.8, where=blocked, color="grey", alpha=0.35)
        ax.text(0.5, 0.5, "Signal paused - patient moving", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#37474f")

    ax.set_xlim(-WINDOW_SEC, 0)
    ax.set_ylim(-1.8, 1.8)
    ax.set_xlabel("Seconds (live)")
    ax.set_ylabel("Chest motion")
    ax.set_title("Radar Chest-Wall Motion Signal (simulated)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ============================================================
# UI
# ============================================================

st.title("📡 VitoRadar Live Patient Monitor")
st.caption("Contactless vital-signs monitoring using Ku-band sensing - "
           "simulated radar signal for the software demo")

col_a, col_b = st.columns(2)
patient = col_a.text_input("Patient name", "Demo Patient")
bed = col_b.text_input("Bed no.", "3")

start = st.button("▶ Start Live Monitoring", type="primary")

status_ph = st.empty()
m1, m2, m3 = st.columns(3)
br_ph, hr_ph, t_ph = m1.empty(), m2.empty(), m3.empty()
chart_ph = st.empty()

st.subheader("🏥 Nurse Station")
nurse_ph = st.empty()
audio_ph = st.empty()


def render_nurse(msgs):
    with nurse_ph.container():
        if not msgs:
            st.caption("No alerts yet.")
        for kind, text in msgs[::-1]:
            if kind == "critical":
                st.error(text)
            elif kind == "warn":
                st.warning(text)
            else:
                st.success(text)


def show_status(status):
    if status == "CALIBRATING":
        status_ph.info("⏳ Calibrating - collecting radar signal...")
    elif status == "NORMAL":
        status_ph.success("✅ NORMAL - breathing and heart rate are in the normal range")
    elif status == "IRREGULAR":
        status_ph.warning("⚠️ IRREGULAR BREATHING - rate outside 10-22 breaths/min")
    elif status == "MOTION":
        status_ph.info("🚶 MOTION DETECTED - patient is moving. "
                       "Radar signal paused, readings on hold")
    else:
        status_ph.error("🚨 APNEA ALERT - breathing has stopped! Nurse station notified")


if not start:
    status_ph.info("👆 Press **Start Live Monitoring**. Turn the volume up for the alarm.")
    render_nurse([])
    st.markdown(
        """
        **Demo story (one patient, ~75 seconds):**
        Normal breathing and heart rate → Irregular breathing → Patient moves
        (radar signal paused) → Back to normal → Sudden apnea → Alarm + message to nurse station.

        **Pipeline:** Antenna → RF Detector → Filter → ESP32 → FFT → Analysis → Dashboard + Nurse alert
        """
    )
    st.warning("Note: signals here are simulated. Only the Ku-band antenna is built in hardware; "
               "RF detector, filter and ESP32 are the next stage.")
else:
    valid = np.array([])        # samples that were accepted (no motion)
    disp = []                   # samples for the graph (NaN = blocked)
    msgs = []                   # nurse station messages
    render_nurse(msgs)

    last_br, last_hr = None, None
    prev_status = None
    hold = 0
    sample_idx = 0
    frame = 0
    alert_frame = None
    responded = False

    for stage, seconds in TIMELINE:
        for _ in range(int(seconds * FS / STEP)):
            t = (np.arange(STEP) + sample_idx) / FS
            sample_idx += STEP
            chunk = radar_samples(stage, t)

            # motion check: block the radar signal while the patient moves
            if is_motion(chunk):
                hold = 4
            blocked = hold > 0
            if blocked:
                hold -= 1
                disp.extend([np.nan] * STEP)
            else:
                disp.extend(chunk.tolist())
                valid = np.concatenate([valid, chunk])

            # analysis (only on accepted signal)
            apnea = False
            if not blocked:
                apnea = is_apnea(valid)
                br = breathing_rate(valid)
                hr = heart_rate(valid)
                if br is not None:
                    last_br = 0.0 if apnea else br
                if hr is not None:
                    last_hr = hr

            # decide status
            if blocked:
                status = "MOTION"
            elif last_br is None:
                status = "CALIBRATING"
            elif apnea:
                status = "APNEA"
            elif last_br > 22 or last_br < 10:
                status = "IRREGULAR"
            else:
                status = "NORMAL"

            # events -> nurse station
            now = datetime.now().strftime("%H:%M:%S")
            if status != prev_status:
                if status == "IRREGULAR":
                    msgs.append(("warn", f"⚠️ {now} | Bed {bed} - {patient}: "
                                         f"irregular breathing ({last_br:.0f}/min). Please check."))
                    render_nurse(msgs)
                elif status == "APNEA":
                    msgs.append(("critical", f"🚨 {now} | Bed {bed} - {patient}: "
                                             f"APNEA - breathing stopped! Immediate attention required."))
                    render_nurse(msgs)
                    alert_frame = frame
                    try:
                        audio_ph.audio(make_siren(), format="audio/wav", autoplay=True)
                    except TypeError:
                        audio_ph.audio(make_siren(), format="audio/wav")
                prev_status = status

            if alert_frame is not None and not responded and frame - alert_frame >= 10:
                msgs.append(("ok", f"✅ {datetime.now().strftime('%H:%M:%S')} | "
                                   f"Nurse responding to Bed {bed} (simulated)"))
                render_nurse(msgs)
                responded = True

            # draw
            show_status(status)
            br_ph.metric("Breathing rate",
                         "--" if last_br is None else f"{last_br:.0f} /min")
            hr_ph.metric("Heart rate",
                         "--" if last_hr is None else f"{last_hr:.0f} BPM")
            t_ph.metric("Monitoring time", f"{sample_idx // FS} s")

            fig = draw_chart(disp, status)
            chart_ph.pyplot(fig)
            plt.close(fig)

            frame += 1
            time.sleep(0.2)

    st.caption("Demo finished. Press **Start Live Monitoring** to run it again.")
