import streamlit as st
from openai import OpenAI
from gradio_client import Client
import json
import re
import io
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AuraTune DSP | AI Music & Acoustic Engineering Studio",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "song_data" not in st.session_state:
    st.session_state.song_data = None
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

# ==========================================
# 2. CUSTOM WEBAPP CSS (DARK STUDIO & TELEPROMPTER UI)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 10%, #161130 0%, #090b10 50%, #050608 100%);
        color: #f0f4f8;
    }

    [data-testid="stSidebar"] {
        background: rgba(11, 13, 20, 0.92) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(0, 242, 254, 0.12);
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 45%, #b465da 85%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        letter-spacing: -0.02em;
    }

    .hero-sub {
        color: #94a3b8;
        font-size: 0.98rem;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 1.5rem;
    }

    /* Glassmorphic Telemetry Cards */
    .studio-card {
        background: rgba(18, 22, 34, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 3px solid #00f2fe;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 8px 28px 0 rgba(0, 0, 0, 0.45);
        margin-bottom: 12px;
    }
    .card-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #94a3b8;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .card-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
    }
    .card-sub {
        font-size: 0.78rem;
        color: #00f2fe;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Technical DSP Micro-Cards */
    .dsp-box {
        background: rgba(10, 14, 23, 0.9);
        border: 1px solid rgba(180, 101, 218, 0.28);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }
    .dsp-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.2rem;
        font-weight: 700;
        color: #00f2fe;
    }
    .dsp-lbl {
        font-size: 0.7rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Custom Lyrics Studio Teleprompter */
    .lyric-container {
        background: rgba(12, 15, 24, 0.88);
        border: 1px solid rgba(0, 242, 254, 0.22);
        border-radius: 14px;
        padding: 22px;
        height: 490px;
        overflow-y: auto;
        font-family: 'Inter', sans-serif;
        line-height: 1.75;
        color: #e2e8f0;
        box-shadow: inset 0 0 25px rgba(0, 0, 0, 0.6);
    }
    .lyric-tag {
        display: inline-block;
        background: rgba(0, 242, 254, 0.15);
        color: #00f2fe;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 6px;
        border: 1px solid rgba(0, 242, 254, 0.35);
        margin-top: 14px;
        margin-bottom: 6px;
        text-transform: uppercase;
    }

    .tag-pill {
        display: inline-block;
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.12), rgba(180, 101, 218, 0.12));
        border: 1px solid rgba(0, 242, 254, 0.3);
        color: #e2e8f0;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 50%, #9b51e0 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        font-size: 1.02rem;
        box-shadow: 0 4px 20px rgba(0, 198, 255, 0.35);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR: STUDIO RACK
# ==========================================
with st.sidebar:
    st.markdown("### 🎛️ DSP & AI RACK")
    api_key = st.text_input("🔑 OpenRouter API Key", type="password", placeholder="sk-or-v1-...")
    if api_key:
        st.success("⚡ Neural Engine Online")
    
    st.divider()
    st.markdown("#### 🎸 Acoustic Parameters")
    genre = st.selectbox(
        "Musical Genre",
        ["Blues Rock", "Alternative Rock", "Progressive Metal", "Indie City Pop", "Acoustic Ballad", "Lo-Fi Chillhop", "Dark Synthwave"]
    )
    instruments = st.multiselect(
        "Lead Instruments",
        ["Electric Guitar", "Clean Stratocaster", "Acoustic Guitar", "Bass Guitar", "Drums", "Synthesizer", "Piano"],
        default=["Electric Guitar", "Bass Guitar", "Drums"]
    )
    vocal_style = st.selectbox(
        "🎙️ Vocal Timbre Profile",
        [
            "male vocal, soulful, expressive, raspy rock singing",
            "male vocal, energetic, melodic, clear rock singing",
            "female vocal, warm, emotional, soulful singing",
            "female vocal, bright, upbeat, pop singing",
            "instrumental only (no vocals)"
        ]
    )
    song_duration = st.slider("⏱️ Synthesis Window (Seconds)", min_value=30, max_value=120, value=60, step=15)

# ==========================================
# 4. BULLETPROOF AI SONGWRITER & EMOTION ENGINE
# ==========================================
def generate_song_plan(key, prompt, genre, instruments, vocal_vibe):
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key.strip())
    
    system_prompt = """You are an elite AI Music Producer, Psychoacoustic Analyst, and Songwriter.
    Return ONLY a valid JSON object with these exact keys:
    {
      "primary_emotion": "e.g., Euphoric Freedom",
      "emotion_score": 94,
      "secondary_emotion": "e.g., Midnight Nostalgia",
      "emotion_vector": {"Valence": 82, "Arousal": 90, "Tension": 65, "Melancholy": 45, "Drive": 92, "Warmth": 78},
      "title": "Creative Song Title",
      "key_signature": "e.g., E Minor",
      "bpm": 122,
      "chords": "e.g., Em - G - D - A",
      "style_tags": ["Blues Rock", "Soulful Vocals", "Overdriven Guitar", "Studio Mix"],
      "lyrics_lines": [
        "[verse]",
        "First line of verse one",
        "Second line of verse one",
        "Third line of verse one",
        "Fourth line of verse one",
        "[chorus]",
        "First line of chorus",
        "Second line of chorus",
        "Third line of chorus",
        "Fourth line of chorus",
        "[bridge]",
        "First line of bridge",
        "Second line of bridge",
        "[outro]",
        "Final fading line"
      ]
    }
    CRITICAL: Put the lyrics as an array of strings inside "lyrics_lines" so JSON never breaks. Write original, poetic lyrics matching the user's story!"""

    user_msg = f"Story/Mood: {prompt}\nGenre: {genre}\nInstruments: {', '.join(instruments)}\nVocal Style: {vocal_vibe}"

    models = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-3-27b-it:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "openai/gpt-4o-mini"
    ]

    last_err = ""
    for m in models:
        try:
            res = client.chat.completions.create(
                model=m,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
                temperature=0.7
            )
            raw = res.choices[0].message.content.strip()
            # Extract JSON block safely
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group(0)
            data = json.loads(raw)
            
            # Guarantee lyrics string is always built cleanly
            lines = data.get("lyrics_lines", [])
            if isinstance(lines, list) and len(lines) > 0:
                data["lyrics"] = "\n".join(lines)
            elif not data.get("lyrics"):
                data["lyrics"] = f"[verse]\n{prompt}\n\n[chorus]\nFeeling the {data.get('primary_emotion', 'rhythm')} tonight"
            return data
        except Exception as e:
            last_err = str(e)
            continue
    raise Exception(f"API Error: {last_err}")

# ==========================================
# 5. ACE-STEP 3.5B NEURAL AUDIO SYNTHESIS
# ==========================================
def generate_real_ai_song_acestep(style_prompt, lyrics_text, duration_sec, is_instrumental=False):
    client = Client("ACE-Step/ACE-Step")
    final_lyrics = "[instrumental]" if is_instrumental else lyrics_text
    
    result = client.predict(
        audio_duration=float(duration_sec),
        prompt=style_prompt,
        lyrics=final_lyrics,
        infer_step=60,
        guidance_scale=15,
        scheduler_type="euler",
        cfg_type="apg",
        omega_scale=10,
        manual_seeds=None,
        guidance_interval=0.5,
        guidance_interval_decay=0,
        min_guidance_scale=3,
        use_erg_tag=True,
        use_erg_lyric=True,
        use_erg_diffusion=True,
        oss_steps=None,
        guidance_scale_text=0,
        guidance_scale_lyric=0,
        audio2audio_enable=False,
        ref_audio_strength=0.5,
        ref_audio_input=None,
        lora_name_or_path="none",
        api_name="/__call__"
    )
    audio_filepath = result[0] if isinstance(result, (list, tuple)) else result
    with open(audio_filepath, "rb") as f:
        return f.read()

# ==========================================
# 6. ADVANCED DSP & PSYCHOACOUSTIC TELEMETRY SUITE
# ==========================================
def analyze_and_plot_dsp_suite(wav_bytes, emotion_vector):
    """
    Runs deep Digital Signal Processing (DSP) analysis on the generated audio track:
    1. Waveform + Dynamic RMS Envelope
    2. 12-Tone Chromagram (Pitch Class Energy)
    3. Harmonic vs. Percussive Source Separation (HPSS)
    4. Spectral Centroid & Rolloff Bandwidth
    Plus a 6-Axis Psychoacoustic Radar Chart!
    """
    y, sr = librosa.load(io.BytesIO(wav_bytes), duration=60)
    
    # 1. Core Signal Telemetry
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    
    # 2. Harmonic-Percussive Source Separation (HPSS)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    rms_harm = librosa.feature.rms(y=y_harmonic)[0]
    rms_perc = librosa.feature.rms(y=y_percussive)[0]
    
    # 3. 12-Tone Chromagram
    chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
    
    # 4. Mastering Engineering Metrics
    peak_amp = float(np.max(np.abs(y)))
    rms_mean = float(np.mean(rms))
    lufs_est = round(20 * np.log10(rms_mean + 1e-9) - 0.691, 1)
    crest_db = round(20 * np.log10((peak_amp / (rms_mean + 1e-9)) + 1e-9), 1)
    tempo_est, _ = librosa.beat.beat_track(y=y_percussive, sr=sr)
    detected_bpm = round(float(np.ndim(tempo_est) and tempo_est[0] or tempo_est), 1)

    # --- BUILD 4-PANEL OSCILLOSCOPE FIGURE ---
    plt.style.use("dark_background")
    fig_dsp, axes = plt.subplots(2, 2, figsize=(14, 7.5), facecolor="#080a10")
    plt.subplots_adjust(hspace=0.38, wspace=0.22)

    # Panel 1: Waveform & Dynamic RMS Envelope
    ax1 = axes[0, 0]
    librosa.display.waveshow(y, sr=sr, ax=ax1, color="#1e90ff", alpha=0.55)
    ax1.plot(times, rms * 2.5, color="#00f2fe", linewidth=1.6, label="RMS Dynamic Envelope")
    ax1.set_title("01 // MASTER WAVEFORM & RMS AMPLITUDE ENVELOPE", fontsize=9.5, color="#00f2fe", fontweight="bold", fontfamily="monospace")
    ax1.set_facecolor("#0d111a")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(color="#1e2638", linestyle="--", linewidth=0.5)

    # Panel 2: 12-Tone Chromagram (Harmonic Pitch Class Distribution)
    ax2 = axes[0, 1]
    img2 = librosa.display.specshow(chroma, y_axis="chroma", x_axis="time", ax=ax2, cmap="cool")
    ax2.set_title("02 // 12-TONE CHROMAGRAM (HARMONIC PITCH ENERGY)", fontsize=9.5, color="#b465da", fontweight="bold", fontfamily="monospace")
    ax2.set_facecolor("#0d111a")
    fig_dsp.colorbar(img2, ax=ax2, format="%0.1f")

    # Panel 3: Harmonic (Vocals/Guitar) vs Percussive (Drums) Separation
    ax3 = axes[1, 0]
    ax3.plot(times, rms_harm, color="#00f2fe", alpha=0.9, linewidth=1.4, label="Harmonic Stem (Vocals/Chords)")
    ax3.plot(times, rms_perc, color="#ff4b72", alpha=0.85, linewidth=1.4, label="Percussive Transients (Drums)")
    ax3.fill_between(times, rms_harm, color="#00f2fe", alpha=0.12)
    ax3.fill_between(times, rms_perc, color="#ff4b72", alpha=0.12)
    ax3.set_title("03 // HPSS STEM DECOMPOSITION (HARMONIC VS. PERCUSSIVE)", fontsize=9.5, color="#ff6b6b", fontweight="bold", fontfamily="monospace")
    ax3.set_facecolor("#0d111a")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(color="#1e2638", linestyle="--", linewidth=0.5)

    # Panel 4: Spectral Centroid & 85% Rolloff Frequency
    ax4 = axes[1, 1]
    ax4.plot(times, cent, color="#facc15", linewidth=1.3, label="Spectral Centroid (Timbral Brightness)")
    ax4.plot(times, rolloff, color="#a855f7", linewidth=1.1, linestyle="--", label="85% Spectral Rolloff")
    ax4.set_title("04 // SPECTRAL TIMBRE & FREQUENCY BANDWIDTH (Hz)", fontsize=9.5, color="#facc15", fontweight="bold", fontfamily="monospace")
    ax4.set_ylabel("Frequency (Hz)")
    ax4.set_facecolor("#0d111a")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(color="#1e2638", linestyle="--", linewidth=0.5)

    # --- BUILD 6-AXIS PSYCHOACOUSTIC RADAR CHART ---
    if not isinstance(emotion_vector, dict) or len(emotion_vector) < 4:
        emotion_vector = {"Valence": 78, "Arousal": 88, "Tension": 64, "Melancholy": 52, "Drive": 91, "Warmth": 80}
    
    labels = list(emotion_vector.keys())
    values = [float(v) for v in emotion_vector.values()]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig_radar, ax_r = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True), facecolor="#080a10")
    ax_r.set_facecolor("#0d111a")
    ax_r.plot(angles, values, color="#00f2fe", linewidth=2.2)
    ax_r.fill(angles, values, color="#00f2fe", alpha=0.25)
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(labels, color="#e2e8f0", fontsize=9, fontfamily="monospace", fontweight="bold")
    ax_r.set_yticklabels([])
    ax_r.grid(color="#263044", linestyle="--")
    ax_r.spines["polar"].set_color("#00f2fe")
    ax_r.set_title("PSYCHOACOUSTIC VECTOR", color="#00f2fe", fontsize=10, fontfamily="monospace", fontweight="bold", pad=15)

    metrics = {
        "lufs": f"{lufs_est} LUFS",
        "crest": f"{crest_db} dB",
        "centroid": f"{int(np.mean(cent))} Hz",
        "zcr": f"{round(float(np.mean(zcr)), 4)}",
        "detected_bpm": f"{detected_bpm} BPM"
    }
    return fig_dsp, fig_radar, metrics

def format_lyrics_html(lyrics_str):
    """Formats lyrics into a glowing studio teleprompter box so they ALWAYS show cleanly."""
    if not lyrics_str or not lyrics_str.strip():
        return "<div class='lyric-container'>No lyrics generated.</div>"
    
    html_lines = ["<div class='lyric-container'>"]
    for line in lyrics_str.split("\n"):
        clean = line.strip()
        if not clean:
            html_lines.append("<br>")
        elif clean.startswith("[") and clean.endswith("]"):
            html_lines.append(f"<div><span class='lyric-tag'>{clean}</span></div>")
        else:
            html_lines.append(f"<div>{clean}</div>")
    html_lines.append("</div>")
    return "".join(html_lines)

# ==========================================
# 7. MAIN WORKSPACE
# ==========================================
st.markdown('<div class="hero-title">🎛️ AuraTune AI // DSP & Neural Music Workstation</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">SYS_PIPELINE: LLM_LYRIC_SYNTH -> ACE_STEP_3.5B_DIFFUSION -> LIBROSA_HPSS_CHROMAGRAM_TELEMETRY</div>', unsafe_allow_html=True)

user_prompt = st.text_area(
    "✨ Input Narrative / Emotional Seed Prompt:",
    placeholder="e.g., Standing under the stage lights in an empty venue after everyone else has gone home, plugging my guitar straight into a cranked tube amp...",
    height=105
)

if st.button("🚀 EXECUTE PIPELINE: SYNTHESIZE VOCALS, BAND & DSP TELEMETRY", use_container_width=True):
    if not api_key:
        st.error("⚠️ Paste your OpenRouter API key in the left sidebar first!")
    elif not user_prompt.strip():
        st.warning("⚠️ Enter a story or mood concept above first!")
    else:
        with st.spinner("🧠 [STAGE 1/3] Computing Psychoacoustic Emotion Vector, Harmonic Chords & Song Lyrics..."):
            try:
                st.session_state.song_data = generate_song_plan(api_key, user_prompt, genre, instruments, vocal_style)
            except Exception as e:
                st.error(str(e))
                st.stop()

        data = st.session_state.song_data
        style_prompt = f"{genre}, {', '.join(instruments)}, {vocal_style}, {data.get('primary_emotion', 'emotional')}, {data.get('bpm', 110)} BPM, studio master, high fidelity"
        is_inst = "instrumental" in vocal_style.lower()

        with st.spinner("🎙️ [STAGE 2/3] ACE-Step 3.5B Diffusion Engine synthesizing Singing Vocals & Multi-Track Band (~35-55s)..."):
            try:
                st.session_state.audio_bytes = generate_real_ai_song_acestep(
                    style_prompt=style_prompt,
                    lyrics_text=data.get("lyrics", ""),
                    duration_sec=song_duration,
                    is_instrumental=is_inst
                )
            except Exception as e:
                st.error(f"Cloud GPU Queue Busy: {e}. Wait 15 seconds and click Execute again!")

# ==========================================
# 8. STUDIO OUTPUT & TECHNICAL TELEMETRY
# ==========================================
if st.session_state.song_data:
    data = st.session_state.song_data
    st.divider()
    
    st.markdown(f"## 💿 MASTER TRACK // `{data.get('title', 'UNTITLED').upper()}`")
    tags = data.get("style_tags", [genre] + instruments[:2])
    tags_html = "".join([f'<span class="tag-pill">{t}</span>' for t in tags])
    st.markdown(tags_html, unsafe_allow_html=True)

    # Row 1: Primary Music Theory & Emotion Telemetry
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="studio-card">
            <div class="card-label">🎭 Dominant Emotion</div>
            <div class="card-value">{data.get('primary_emotion', 'Euphoric')}</div>
            <div class="card-sub">CONFIDENCE: {data.get('emotion_score', 94)}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="studio-card">
            <div class="card-label">🎹 Harmonic Key</div>
            <div class="card-value">{data.get('key_signature', 'E Minor')}</div>
            <div class="card-sub">UNDERTONE: {data.get('secondary_emotion', 'Reflective')}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="studio-card">
            <div class="card-label">⏱️ Target Tempo</div>
            <div class="card-value">{data.get('bpm', 120)} BPM</div>
            <div class="card-sub">METER: 4/4 STANDARD</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="studio-card">
            <div class="card-label">🎸 Chord Voicing</div>
            <div class="card-value">{data.get('chords', 'Em - G - D - A')}</div>
            <div class="card-sub">DIATONIC PROGRESSION</div>
        </div>
        """, unsafe_allow_html=True)

    # Row 2: Master Playback + Radar Vector + Guaranteed Lyrics Teleprompter
    col_left, col_mid, col_right = st.columns([1.4, 1.1, 1.5], gap="medium")

    if st.session_state.audio_bytes:
        fig_dsp, fig_radar, dsp_metrics = analyze_and_plot_dsp_suite(
            st.session_state.audio_bytes,
            data.get("emotion_vector", {})
        )

        with col_left:
            st.markdown("#### 🎧 Neural Master Bus")
            st.audio(st.session_state.audio_bytes, format="audio/wav")
            st.download_button(
                "⬇️ EXPORT 44.1kHz MASTER (.WAV)",
                data=st.session_state.audio_bytes,
                file_name=f"{data.get('title', 'song').replace(' ', '_')}_Master.wav",
                mime="audio/wav",
                use_container_width=True
            )
            
            st.write("")
            st.markdown("#### 🎚️ Mastering & Signal Readout")
            m_a, m_b = st.columns(2)
            m_a.markdown(f"<div class='dsp-box'><div class='dsp-val'>{dsp_metrics['lufs']}</div><div class='dsp-lbl'>Integrated Loudness</div></div>", unsafe_allow_html=True)
            m_b.markdown(f"<div class='dsp-box'><div class='dsp-val'>{dsp_metrics['crest']}</div><div class='dsp-lbl'>Dynamic Crest Factor</div></div>", unsafe_allow_html=True)
            st.write("")
            m_c, m_d = st.columns(2)
            m_c.markdown(f"<div class='dsp-box'><div class='dsp-val'>{dsp_metrics['centroid']}</div><div class='dsp-lbl'>Spectral Centroid</div></div>", unsafe_allow_html=True)
            m_d.markdown(f"<div class='dsp-box'><div class='dsp-val'>{dsp_metrics['zcr']}</div><div class='dsp-lbl'>Zero-Crossing Rate</div></div>", unsafe_allow_html=True)

        with col_mid:
            st.markdown("#### 🕸️ Psychoacoustic Vector")
            st.pyplot(fig_radar)

    with col_right:
        st.markdown("#### 📝 Vocal Teleprompter & Structure")
        st.markdown(format_lyrics_html(data.get("lyrics", "")), unsafe_allow_html=True)

    # Row 3: Full-Width 4-Panel Acoustic Engineering Oscilloscope Suite
    if st.session_state.audio_bytes:
        st.write("")
        st.markdown("### 🔬 Multi-Track DSP & Spectral Decomposition Suite (`Librosa HPSS + STFT`)")
        st.pyplot(fig_dsp)