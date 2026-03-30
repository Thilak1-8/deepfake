import { useState, useRef } from 'react';
import './index.css';

// Using raw fetch for simplicity, configure base API url based on env
const API_URL = 'http://localhost:5000';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);
  const analyzeSectionRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragging(true);
    } else if (e.type === "dragleave") {
      setIsDragging(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (selectedFile) => {
    if (!selectedFile.type.startsWith('image/')) {
      setError('Please upload a valid image file (PNG, JPG, JPEG)');
      return;
    }

    setError(null);
    setFile(selectedFile);

    const reader = new FileReader();
    reader.onload = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(selectedFile);

    setResults(null);
  };

  const analyzeImage = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'An error occurred during analysis');
      }

      setTimeout(() => {
        setResults(data);
        setIsAnalyzing(false);
      }, 600);

    } catch (err) {
      console.error(err);
      setError(err.message || 'Failed to connect to the server. Is the Python backend running?');
      setIsAnalyzing(false);
    }
  };

  const resetAll = () => {
    setFile(null);
    setPreview(null);
    setResults(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const scrollToAnalyze = (e) => {
    e.preventDefault();
    analyzeSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <>
      {/* NAV */}
      <nav>
        <div className="logo">Deep<span>Shield</span></div>
        <ul className="nav-links">
          <li><a href="#how">How It Works</a></li>
          <li><a href="#threat">Threat Data</a></li>
          <li><a href="#technology">Technology</a></li>
          <li><a href="#news">News</a></li>
        </ul>
        <button className="nav-cta" onClick={scrollToAnalyze}>Analyze Image</button>
      </nav>

      {/* HERO */}
      <section>
        <div className="hero">
          <div className="hero-badge">AI-powered detection system</div>
          <h1>Deepfakes are<br />real. <em>Detection</em><br />is too.</h1>
          <p className="hero-sub">
            Our 6-channel forensic AI analyzes every pixel — from frequency artifacts to wavelet patterns — to expose synthetic media with industry-leading precision.
          </p>
          <div className="hero-actions">
            <a href="#analyze" className="btn-primary" onClick={scrollToAnalyze}>Detect an Image</a>
            <a href="#how" className="btn-secondary">See How It Works</a>
          </div>

          <div className="acc-bar-wrap" style={{ maxWidth: '460px', marginTop: '3rem' }}>
            <div className="acc-bar-label">
              <span>Test Accuracy</span>
              <span style={{ color: 'var(--accent)' }}>93.64%</span>
            </div>
            <div className="acc-bar"><div className="acc-bar-fill" style={{ width: '93.64%' }}></div></div>

            <div className="acc-bar-label" style={{ marginTop: '1rem' }}>
              <span>ROC AUC Score</span>
              <span style={{ color: 'var(--accent)' }}>98.40%</span>
            </div>
            <div className="acc-bar"><div className="acc-bar-fill" style={{ width: '98.4%' }}></div></div>

            <div className="acc-bar-label" style={{ marginTop: '1rem' }}>
              <span>F1-Score</span>
              <span style={{ color: 'var(--accent3)' }}>92.86%</span>
            </div>
            <div className="acc-bar"><div className="acc-bar-fill" style={{ width: '92.86%', background: 'var(--accent3)' }}></div></div>

            <div className="acc-bar-label" style={{ marginTop: '1rem' }}>
              <span>Human Detection Rate</span>
              <span style={{ color: 'var(--danger)' }}>24.5%</span>
            </div>
            <div className="acc-bar"><div className="acc-bar-fill" style={{ width: '24.5%', background: 'var(--danger)' }}></div></div>
          </div>
        </div>
      </section>

      {/* TICKER */}
      <div className="ticker-wrap" role="marquee" aria-label="Live threat statistics">
        <div className="ticker-track" id="ticker">
          <span className="ticker-item"><span className="dot"></span> Deepfake attempt every <span className="num">5 minutes</span> in 2024</span>
          <span className="ticker-item"><span className="dot"></span> <span className="num">900%</span> projected annual growth in deepfake volume</span>
          <span className="ticker-item"><span className="dot"></span> <span className="num">$25M</span> stolen in single deepfake video call — Arup 2024</span>
          <span className="ticker-item"><span className="dot"></span> Only <span className="num">0.1%</span> of people correctly identified all fake &amp; real media</span>
          <span className="ticker-item"><span className="dot"></span> <span className="num">8M+</span> deepfake files projected by end of 2025</span>
          <span className="ticker-item"><span className="dot"></span> <span className="num">3,000%</span> surge in deepfake identity fraud in 2023</span>
          <span className="ticker-item"><span className="dot"></span> <span className="num">$40B</span> AI fraud losses projected in US by 2027</span>
          <span className="ticker-item"><span className="dot"></span> Deepfake attempt every <span className="num">5 minutes</span> in 2024</span>
          <span className="ticker-item"><span className="dot"></span> <span className="num">900%</span> projected annual growth in deepfake volume</span>
          <span className="ticker-item"><span className="dot"></span> <span className="num">$25M</span> stolen in single deepfake video call — Arup 2024</span>
          <span className="ticker-item"><span className="dot"></span> Only <span className="num">0.1%</span> of people correctly identified all fake &amp; real media</span>
          <span className="ticker-item"><span className="dot"></span> <span className="num">8M+</span> deepfake files projected by end of 2025</span>
          <span className="ticker-item"><span className="dot"></span> <span className="num">3,000%</span> surge in deepfake identity fraud in 2023</span>
          <span className="ticker-item"><span className="dot"></span> <span className="num">$40B</span> AI fraud losses projected in US by 2027</span>
        </div>
      </div>

      {/* STATS GRID */}
      <section className="stats-section" id="threat">
        <div className="section-label">Model Performance</div>
        <h2 className="section-title">Real metrics.<br />Real results.</h2>

        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-num">93.64<span className="stat-unit">%</span></div>
            <div className="stat-label">test set accuracy on 1,934 held-out images — Fake &amp; Real</div>
            <div className="stat-source">// ResNet50 · 6-channel model · test split</div>
          </div>
          <div className="stat-card">
            <div className="stat-num">98.40<span className="stat-unit">%</span></div>
            <div className="stat-label">ROC AUC score — near-perfect class separability</div>
            <div className="stat-source">// test set evaluation</div>
          </div>
          <div className="stat-card">
            <div className="stat-num">92.86<span className="stat-unit">%</span></div>
            <div className="stat-label">F1-Score across both Fake and Real classes (weighted avg)</div>
            <div className="stat-source">// precision 93.79% · recall 91.95%</div>
          </div>
          <div className="stat-card">
            <div className="stat-num">12,890</div>
            <div className="stat-label">total training images — 5,890 real faces + 7,000 fake faces</div>
            <div className="stat-source">// class-weighted to handle imbalance</div>
          </div>
          <div className="stat-card">
            <div className="stat-num">25.6<span className="stat-unit">M</span></div>
            <div className="stat-label">trainable parameters in the modified ResNet50 architecture</div>
            <div className="stat-source">// 6-channel input · custom classification head</div>
          </div>
          <div className="stat-card">
            <div className="stat-num">6</div>
            <div className="stat-label">forensic feature channels: Y · Cb · Cr · FFT · DCT · Wavelet</div>
            <div className="stat-source">// spatial + frequency domain fusion</div>
          </div>
        </div>
      </section>

      <div className="divider"></div>

      {/* THREATS */}
      <section className="threat-section">
        <div className="threat-inner">
          <div className="section-label">Attack Vectors</div>
          <h2 className="section-title">Three ways<br />deepfakes strike.</h2>

          <div className="threat-grid">
            <div className="threat-card">
              <div className="threat-icon">📣</div>
              <h3>Content Propaganda</h3>
              <p>AI-generated political disinformation spreads through social platforms at exponential speed — false narratives reach up to 100,000 people while truth rarely hits 1,000.</p>
              <div className="threat-stat">↑ 1,000% spear phishing increase / decade</div>
            </div>
            <div className="threat-card">
              <div className="threat-icon">🎭</div>
              <h3>Identity Fraud &amp; Fake Profiles</h3>
              <p>Synthetic faces impersonate executives, colleagues, and public figures on video calls and social platforms — 400+ companies face CEO deepfake scams daily.</p>
              <div className="threat-stat">↑ 1,740% increase in North America</div>
            </div>
            <div className="threat-card">
              <div className="threat-icon">🎥</div>
              <h3>Real-Time Video Manipulation</h3>
              <p>Live face-swap technology enables fraudsters to pass biometric verification in real-time — the same tech that stole $25M from Arup in a single conference call.</p>
              <div className="threat-stat">↑ Attack every 5 minutes in 2024</div>
            </div>
          </div>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section className="how-section" id="how">
        <div className="section-label">Detection Pipeline</div>
        <h2 className="section-title">Six channels.<br />One verdict.</h2>

        <div className="pipeline">
          <div className="pipe-step">
            <span className="pipe-num">// CH.01</span>
            <div className="pipe-icon-wrap">Y</div>
            <h3>Color Space (YCbCr)</h3>
            <p>Separates luminance from chrominance. GAN artifacts hide in Cb/Cr channels invisible to the human eye.</p>
          </div>
          <div className="pipe-step">
            <span className="pipe-num">// CH.02</span>
            <div className="pipe-icon-wrap">∿</div>
            <h3>FFT Magnitude</h3>
            <p>2D Fast Fourier Transform on the Y channel exposes periodic frequency artifacts and compression ghosts.</p>
          </div>
          <div className="pipe-step">
            <span className="pipe-num">// CH.03</span>
            <div className="pipe-icon-wrap">Σ</div>
            <h3>DCT Coefficients</h3>
            <p>Discrete Cosine Transform reveals JPEG-layer inconsistencies left by generative model upsampling.</p>
          </div>
          <div className="pipe-step">
            <span className="pipe-num">// CH.04</span>
            <div className="pipe-icon-wrap">≋</div>
            <h3>Wavelet (Haar)</h3>
            <p>2D Haar wavelet captures edge and texture irregularities — the subtle tells that break spatial coherence.</p>
          </div>
          <div className="pipe-step">
            <span className="pipe-num">// CH.05</span>
            <div className="pipe-icon-wrap">⊕</div>
            <h3>Feature Fusion</h3>
            <p>All 6 channels stacked into a unified tensor: [Y, Cb, Cr, FFT, DCT, Wavelet] for joint classification.</p>
          </div>
          <div className="pipe-step">
            <span className="pipe-num">// CH.06</span>
            <div className="pipe-icon-wrap">◉</div>
            <h3>Verdict + Score</h3>
            <p>Binary classification with confidence score — REAL or FAKE with per-channel attribution explainability.</p>
          </div>
        </div>
      </section>

      {/* 6 CHANNELS DETAIL */}
      <section className="channels-section" id="technology">
        <div className="channels-inner">
          <div className="section-label">Technical Breakdown</div>
          <h2 className="section-title">Inside the<br />6-channel engine.</h2>

          <div className="channels-grid">
            <div className="channel-card">
              <div className="channel-index">01</div>
              <div className="channel-content">
                <span className="channel-tag">Color Space Transformation</span>
                <h3>YCbCr Analysis</h3>
                <p>RGB conversion to YCbCr decouples luminance from color information. Deepfake generators consistently leave artifacts in the Cb and Cr channels that are imperceptible visually but statistically significant.</p>
              </div>
            </div>
            <div className="channel-card">
              <div className="channel-index">02</div>
              <div className="channel-content">
                <span className="channel-tag">Frequency Domain</span>
                <h3>FFT Magnitude Spectrum</h3>
                <p>2D Fast Fourier Transform on the Y luminance channel moves analysis into the frequency domain. Log-scaled and normalized for stability. Reveals periodic noise patterns and upsampling grid artifacts invisible in spatial domain.</p>
              </div>
            </div>
            <div className="channel-card">
              <div className="channel-index">03</div>
              <div className="channel-content">
                <span className="channel-tag">Compression Analysis</span>
                <h3>DCT Coefficients</h3>
                <p>Discrete Cosine Transform — the same algorithm at the heart of JPEG compression. Exposes double-compression artifacts and coefficient distribution anomalies characteristic of neural renderer outputs.</p>
              </div>
            </div>
            <div className="channel-card">
              <div className="channel-index">04</div>
              <div className="channel-content">
                <span className="channel-tag">Texture &amp; Edge</span>
                <h3>Haar Wavelet Transform</h3>
                <p>2D Discrete Wavelet Transform isolates high-frequency spatial detail via horizontal detail coefficients (cH). GAN inpainting disrupts natural texture statistics in ways the wavelet domain makes legible.</p>
              </div>
            </div>
            <div className="channel-card">
              <div className="channel-index">05</div>
              <div className="channel-content">
                <span className="channel-tag">Multi-Modal Fusion</span>
                <h3>6-Channel Feature Stack</h3>
                <p>[Y · Cb · Cr · FFT · DCT · Wavelet] — six complementary views of the same image, stacked into a unified tensor that feeds the classification head. No single channel is sufficient; fusion is the signal.</p>
              </div>
            </div>
            <div className="channel-card">
              <div className="channel-index">06</div>
              <div className="channel-content">
                <span className="channel-tag">Explainability</span>
                <h3>Per-Channel Attribution</h3>
                <p>Beyond a binary verdict, the model surfaces which channels drove the decision — giving forensic analysts a traceable, auditable basis for each REAL / FAKE classification with channel-level confidence breakdown.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* BIG STAT + ARCHITECTURE */}
      <section className="bigstat-section" id="news">
        <div className="bigstat-left">
          <div className="num">93.64</div>
          <div className="denom">% accuracy</div>
          <p className="desc">
            Achieved on a held-out test set of 1,934 images — never seen during training. The model was trained on 12,890 images with a 70 / 15 / 15 split. Class weights compensate for the 5,890 real vs 7,000 fake imbalance, ensuring neither class dominates learning.
          </p>

          <div className="acc-bar-wrap" style={{ marginTop: '2rem', maxWidth: '360px' }}>
            <div className="acc-bar-label">
              <span>Fake detection (Recall)</span>
              <span style={{ color: 'var(--accent)' }}>95.02%</span>
            </div>
            <div className="acc-bar"><div className="acc-bar-fill" style={{ width: '95.02%' }}></div></div>

            <div className="acc-bar-label" style={{ marginTop: '0.75rem' }}>
              <span>Real detection (Recall)</span>
              <span style={{ color: 'var(--accent3)' }}>91.95%</span>
            </div>
            <div className="acc-bar"><div className="acc-bar-fill" style={{ width: '91.95%', background: 'var(--accent3)' }}></div></div>

            <div className="acc-bar-label" style={{ marginTop: '0.75rem' }}>
              <span>Human Detection Rate</span>
              <span style={{ color: 'var(--danger)' }}>24.5%</span>
            </div>
            <div className="acc-bar"><div className="acc-bar-fill" style={{ width: '24.5%', background: 'var(--danger)' }}></div></div>
          </div>
        </div>

        <div>
          <div className="section-label">Model Architecture</div>
          <div className="news-list">
            <div className="news-item green">
              <div className="news-dot" style={{ background: 'var(--accent)' }}></div>
              <div className="news-content">
                <div className="news-tag">Base Model</div>
                <div className="news-title">Pretrained ResNet50 — modified first conv layer to accept 6 channels instead of 3. Weights initialized by duplicating pretrained RGB weights.</div>
              </div>
            </div>
            <div className="news-item green">
              <div className="news-dot" style={{ background: 'var(--accent)' }}></div>
              <div className="news-content">
                <div className="news-tag">Classification Head</div>
                <div className="news-title">Dropout(0.5) → Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→2) — heavy regularization prevents overfitting.</div>
              </div>
            </div>
            <div className="news-item green">
              <div className="news-dot" style={{ background: 'var(--accent)' }}></div>
              <div className="news-content">
                <div className="news-tag">Optimizer &amp; Scheduler</div>
                <div className="news-title">Adam (lr=0.0001, weight_decay=1e-4) with ReduceLROnPlateau — factor 0.5, patience 3 epochs.</div>
              </div>
            </div>
            <div className="news-item green">
              <div className="news-dot" style={{ background: 'var(--accent)' }}></div>
              <div className="news-content">
                <div className="news-tag">Early Stopping</div>
                <div className="news-title">Patience of 7 epochs with batch size 32. Training halts automatically when validation loss plateaus.</div>
              </div>
            </div>
            <div className="news-item green">
              <div className="news-dot" style={{ background: 'var(--accent)' }}></div>
              <div className="news-content">
                <div className="news-tag">Class Imbalance</div>
                <div className="news-title">sklearn compute_class_weight applied to CrossEntropyLoss — balances the 5,890 real vs 7,000 fake image disparity during training.</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="divider"></div>

      {/* WHY IT MATTERS */}
      <section className="why-section">
        <div className="why-inner">
          <div className="section-label">Why This Approach Works</div>
          <h2 className="section-title">Multi-domain analysis.<br />One unified verdict.</h2>

          <div className="why-grid">
            <div className="why-card">
              <div className="why-num">70/15/15</div>
              <div className="why-label">Data Split</div>
              <p className="why-desc">Train / Validation / Test — rigorous held-out evaluation on 1,934 images never seen during training.</p>
            </div>
            <div className="why-card">
              <div className="why-num">6ch</div>
              <div className="why-label">Feature Channels</div>
              <p className="why-desc">Multi-domain fusion: Y, Cb, Cr (spatial) + FFT, DCT, Wavelet (frequency) — no single channel is enough.</p>
            </div>
            <div className="why-card">
              <div className="why-num">3s</div>
              <div className="why-label">Audio Clone Threshold</div>
              <p className="why-desc">Scammers need just 3 seconds of audio to clone a voice with 85% match accuracy — detection can't wait.</p>
            </div>
            <div className="why-card">
              <div className="why-num">$25M</div>
              <div className="why-label">Single Incident Loss</div>
              <p className="why-desc">Arup, 2024: a single deepfake video call convinced an employee to wire $25M to fraudsters.</p>
            </div>
          </div>
        </div>
      </section>

      {/* CLASSIFICATION REPORT */}
      <section style={{ padding: '5rem 3rem', maxWidth: '1200px', margin: '0 auto' }}>
        <div className="section-label">Test Set Evaluation</div>
        <h2 className="section-title">Classification<br />report.</h2>

        <div style={{ marginTop: '3rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3rem', alignItems: 'start' }}>
          <div>
            <div style={{ border: '1px solid var(--border2)', overflow: 'hidden' }}>
              <div style={{ background: 'var(--bg2)', padding: '0.75rem 1.5rem', borderBottom: '1px solid var(--border2)' }}>
                <span style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.7rem', color: 'var(--accent)', letterSpacing: '0.1em' }}>// TEST SET · 1,934 IMAGES</span>
              </div>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border2)' }}>
                    <th style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.65rem', color: 'var(--text3)', textAlign: 'left', padding: '0.75rem 1.5rem', fontWeight: 400, letterSpacing: '0.08em', textTransform: 'uppercase' }}>Class</th>
                    <th style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.65rem', color: 'var(--text3)', textAlign: 'right', padding: '0.75rem 1rem', fontWeight: 400, letterSpacing: '0.08em', textTransform: 'uppercase' }}>Precision</th>
                    <th style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.65rem', color: 'var(--text3)', textAlign: 'right', padding: '0.75rem 1rem', fontWeight: 400, letterSpacing: '0.08em', textTransform: 'uppercase' }}>Recall</th>
                    <th style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.65rem', color: 'var(--text3)', textAlign: 'right', padding: '0.75rem 1rem', fontWeight: 400, letterSpacing: '0.08em', textTransform: 'uppercase' }}>F1</th>
                    <th style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.65rem', color: 'var(--text3)', textAlign: 'right', padding: '0.75rem 1.5rem', fontWeight: 400, letterSpacing: '0.08em', textTransform: 'uppercase' }}>Support</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: '1px solid var(--border2)' }}>
                    <td style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.75rem', color: 'var(--danger)', padding: '1rem 1.5rem', fontWeight: 700 }}>Fake</td>
                    <td style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.75rem', color: 'var(--text)', textAlign: 'right', padding: '1rem' }}>93.52%</td>
                    <td style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.75rem', color: 'var(--accent)', textAlign: 'right', padding: '1rem' }}>95.02%</td>
                    <td style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.75rem', color: 'var(--text)', textAlign: 'right', padding: '1rem' }}>94.27%</td>
                    <td style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.75rem', color: 'var(--text3)', textAlign: 'right', padding: '1rem 1.5rem' }}>1,064</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid var(--border2)' }}>
                    <td style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.75rem', color: 'var(--accent3)', padding: '1rem 1.5rem', fontWeight: 700 }}>Real</td>
                    <td style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.75rem', color: 'var(--text)', textAlign: 'right', padding: '1rem' }}>93.79%</td>
                    <td style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.75rem', color: 'var(--accent)', textAlign: 'right', padding: '1rem' }}>91.95%</td>
                    <td style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.75rem', color: 'var(--text)', textAlign: 'right', padding: '1rem' }}>92.86%</td>
                    <td style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.75rem', color: 'var(--text3)', textAlign: 'right', padding: '1rem 1.5rem' }}>870</td>
                  </tr>
                  <tr>
                    <td style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.75rem', color: 'var(--text2)', padding: '1rem 1.5rem' }}>Weighted Avg</td>
                    <td style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.75rem', color: 'var(--accent)', textAlign: 'right', padding: '1rem', fontWeight: 700 }}>93.64%</td>
                    <td style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.75rem', color: 'var(--accent)', textAlign: 'right', padding: '1rem', fontWeight: 700 }}>93.64%</td>
                    <td style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.75rem', color: 'var(--accent)', textAlign: 'right', padding: '1rem', fontWeight: 700 }}>93.63%</td>
                    <td style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.75rem', color: 'var(--text3)', textAlign: 'right', padding: '1rem 1.5rem' }}>1,934</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div style={{ border: '1px solid var(--border)', padding: '1.5rem 2rem', background: 'var(--bg2)' }}>
              <div style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.65rem', color: 'var(--text3)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '0.5rem' }}>Overall Accuracy</div>
              <div style={{ fontSize: '2.5rem', fontWeight: 800, color: 'var(--accent)', letterSpacing: '-0.03em', lineHeight: 1 }}>93.64<span style={{ fontSize: '1.2rem', color: 'var(--text2)' }}>%</span></div>
            </div>
            <div style={{ border: '1px solid var(--border)', padding: '1.5rem 2rem', background: 'var(--bg2)' }}>
              <div style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.65rem', color: 'var(--text3)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '0.5rem' }}>ROC AUC Score</div>
              <div style={{ fontSize: '2.5rem', fontWeight: 800, color: 'var(--accent)', letterSpacing: '-0.03em', lineHeight: 1 }}>98.40<span style={{ fontSize: '1.2rem', color: 'var(--text2)' }}>%</span></div>
            </div>
            <div style={{ border: '1px solid var(--border)', padding: '1.5rem 2rem', background: 'var(--bg2)' }}>
              <div style={{ fontFamily: "'Space Mono', monospace", fontSize: '0.65rem', color: 'var(--text3)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '0.5rem' }}>Total Parameters</div>
              <div style={{ fontSize: '2.5rem', fontWeight: 800, color: 'var(--accent)', letterSpacing: '-0.03em', lineHeight: 1 }}>25.6<span style={{ fontSize: '1.2rem', color: 'var(--text2)' }}>M</span></div>
            </div>
          </div>
        </div>
      </section>

      <div className="divider"></div>

      {/* CTA (Replaced with specific functionality integration) */}
      <section className="cta-section" id="analyze" ref={analyzeSectionRef}>
        <div className="section-label" style={{ justifyContent: 'center' }}>Ready to detect</div>
        <h2>Upload an image.<br />Get the <em style={{ color: 'var(--accent)', fontStyle: 'normal' }}>truth.</em></h2>
        <p>Our 6-channel forensic engine returns a verdict in seconds — with per-channel attribution you can trace and trust.</p>

        {!results && (
          <div style={{ background: 'transparent', padding: 0, border: 'none', boxShadow: 'none' }}>
            {!preview ? (
              <div
                className={`upload-container ${isDragging ? 'drag-active' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current.click()}
              >
                <div className="upload-content">
                  <div className="upload-icon">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                      <polyline points="17 8 12 3 7 8"></polyline>
                      <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                  </div>
                  <div className="upload-text">Drag &amp; Drop an image here</div>
                  <div className="upload-subtext">or click to browse files</div>
                </div>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleChange}
                  style={{ display: 'none' }}
                  accept="image/*"
                />
              </div>
            ) : (
              <div className="image-preview-wrapper" style={{ margin: '2rem auto' }}>
                <img src={preview} alt="Upload preview" className="image-preview" style={{ maxWidth: '100%', borderRadius: '12px' }} />
                <div className="cta-btns" style={{ marginTop: '1.5rem', display: 'flex', gap: '1rem', justifyContent: 'center' }}>
                  <button className="btn-secondary" onClick={resetAll} disabled={isAnalyzing}>
                    Choose Different
                  </button>
                  <button
                    className="btn-primary"
                    onClick={analyzeImage}
                    disabled={isAnalyzing}
                  >
                    {isAnalyzing ? (
                      <div className="loader-container">
                        <div className="loader"></div>
                        <span>Analyzing...</span>
                      </div>
                    ) : 'Analyze for Deepfakes'}
                  </button>
                </div>
              </div>
            )}

            {error && (
              <div className="error-alert">
                <span>{error}</span>
              </div>
            )}
          </div>
        )}

        {results && (
          <div className="results-container">
            <div className={`prediction-banner ${results.prediction === 'Fake' ? 'prediction-fake' : 'prediction-real'}`}>
              <div className="prediction-label">Analysis Complete</div>
              <div className="prediction-text">
                AI predicts this image is
                <span className="prediction-result">{results.prediction}</span>
              </div>
              <div style={{ marginTop: '1rem', fontFamily: "'Space Mono', monospace", fontSize: '0.8rem' }}>
                Confidence Score: <strong style={{ color: 'var(--text)' }}>{(results.confidence * 100).toFixed(2)}%</strong>
              </div>
            </div>

            <h3 className="features-section-title">Extracted Multi-Domain Feature Maps</h3>
            <div className="features-grid">
              <div className="feature-card">
                <div className="feature-title">Original</div>
                <img src={`${API_URL}${results.visualizations.original}`} alt="Original" className="feature-image" />
              </div>
              <div className="feature-card">
                <div className="feature-title">Y-Channel</div>
                <img src={`${API_URL}${results.visualizations.y_channel}`} alt="Y Channel" className="feature-image" />
              </div>
              <div className="feature-card">
                <div className="feature-title">FFT Magnitude</div>
                <img src={`${API_URL}${results.visualizations.fft}`} alt="FFT" className="feature-image" />
              </div>
              <div className="feature-card">
                <div className="feature-title">DCT Coeffs</div>
                <img src={`${API_URL}${results.visualizations.dct}`} alt="DCT" className="feature-image" />
              </div>
              <div className="feature-card">
                <div className="feature-title">Wavelet</div>
                <img src={`${API_URL}${results.visualizations.wavelet}`} alt="Wavelet" className="feature-image" />
              </div>
            </div>

            <div className="cta-btns" style={{ marginTop: '2.5rem' }}>
              <button className="btn-secondary" onClick={resetAll}>
                Scan Another Image
              </button>
            </div>
          </div>
        )}
      </section>

      {/* FOOTER */}
      <footer>
        <p>© 2025 DeepShield — AI Deepfake Detection. Built with 6-channel forensic analysis.</p>
        <ul className="footer-links">
          <li><a href="#">API Docs</a></li>
          <li><a href="#">Privacy</a></li>
          <li><a href="#">Research</a></li>
          <li><a href="#">Contact</a></li>
        </ul>
      </footer>
    </>
  );
}

export default App;
