import { useState, useRef, useEffect } from 'react';
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

      // Small delay for smooth transition effect
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

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">Deepfake Sentinel</h1>
        <p className="app-subtitle">Advanced Multi-Domain Architecture Analysis</p>
      </header>

      <main>
        {!results && (
          <div className="glass-panel">
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
                  <div className="upload-text">Drag & Drop an image here</div>
                  <div className="upload-subtext">or click to browse files</div>
                </div>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleChange}
                  className="upload-input"
                  accept="image/*"
                />
              </div>
            ) : (
              <div className="image-preview-wrapper">
                <div className="image-preview-card">
                  <img src={preview} alt="Upload preview" className="image-preview" />
                </div>
                <div className="action-buttons">
                  <button className="secondary-btn" onClick={resetAll} disabled={isAnalyzing}>
                    Choose Different
                  </button>
                  <button
                    className="analyze-button"
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
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="12" y1="8" x2="12" y2="12"></line>
                  <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                <span>{error}</span>
              </div>
            )}
          </div>
        )}

        {results && (
          <div className="results-container glass-panel">
            <div className={`prediction-banner ${results.prediction === 'Fake' ? 'prediction-fake' : 'prediction-real'}`}>
              <div className="prediction-content">
                <div className="prediction-label">Analysis Complete</div>
                <div className="prediction-text">
                  AI predicts this image is
                  <span className="prediction-result">{results.prediction}</span>
                </div>

                <div className="confidence-section">
                  <div className="confidence-label">
                    Confidence Score: <span className="confidence-percent">{(results.confidence * 100).toFixed(2)}%</span>
                  </div>
                  <div className="confidence-bar-bg">
                    <div
                      className={`confidence-fill ${results.prediction.toLowerCase()}`}
                      style={{ width: `${results.confidence * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>

            <h3 className="features-section-title">Extracted Multi-Domain Feature Maps</h3>

            <div className="features-grid">
              <div className="feature-card">
                <div className="feature-title">Original (Resized)</div>
                <div className="feature-image-container">
                  <img src={`${API_URL}${results.visualizations.original}`} alt="Original" className="feature-image" />
                </div>
              </div>

              <div className="feature-card">
                <div className="feature-title">Y-Channel (YCbCr)</div>
                <div className="feature-image-container">
                  <img src={`${API_URL}${results.visualizations.y_channel}`} alt="Y Channel" className="feature-image" />
                </div>
              </div>

              <div className="feature-card">
                <div className="feature-title">FFT Magnitude</div>
                <div className="feature-image-container">
                  <img src={`${API_URL}${results.visualizations.fft}`} alt="FFT" className="feature-image" />
                </div>
              </div>

              <div className="feature-card">
                <div className="feature-title">DCT Coefficients</div>
                <div className="feature-image-container">
                  <img src={`${API_URL}${results.visualizations.dct}`} alt="DCT" className="feature-image" />
                </div>
              </div>

              <div className="feature-card">
                <div className="feature-title">Wavelet (Haar)</div>
                <div className="feature-image-container">
                  <img src={`${API_URL}${results.visualizations.wavelet}`} alt="Wavelet" className="feature-image" />
                </div>
              </div>
            </div>

            <div style={{ textAlign: 'center', marginTop: '3.5rem' }}>
              <button className="secondary-btn" onClick={resetAll} style={{ padding: '1rem 3rem' }}>
                Scan Another Image
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
