import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "/api";

function assetUrl(path) {
  if (!path) return "";
  return path.startsWith("http") ? path : `${API_URL.replace(/\/api$/, "")}${path}`;
}

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);

  function selectFile(selected) {
    if (!selected) return;
    if (!selected.type.startsWith("image/")) {
      setError("Please choose a JPEG, PNG, or WebP image.");
      return;
    }
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError("");
  }

  function chooseFile(event) {
    selectFile(event.target.files?.[0]);
  }

  function dropFile(event) {
    event.preventDefault();
    setDragging(false);
    selectFile(event.dataTransfer.files?.[0]);
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError("");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch(`${API_URL}/analyze`, { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Analysis failed.");
      setResult(payload);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <header>
        <p className="eyebrow">CHROMATIC SYNC / STYLE INTELLIGENCE</p>
        <h1>Find the right piece<br /><em>for every look.</em></h1>
        <p className="intro">Upload a garment and get its category, dominant color, and compatible styling ideas.</p>
      </header>
      <section className="workspace">
        <label className={`dropzone ${dragging ? "dragging" : ""}`}
          onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={dropFile}>
          {preview ? <img src={preview} alt="Selected garment" /> : <span className="upload-icon">+</span>}
          <strong>{file ? file.name : "Drop a garment image here"}</strong>
          <small>{file ? "Click to choose a different image" : "or click to browse · JPG, PNG, or WebP"}</small>
          <input type="file" accept="image/jpeg,image/png,image/webp" onChange={chooseFile} />
        </label>
        <div className="actions">
          <button onClick={analyze} disabled={!file || loading}>
            {loading ? <><span className="spinner" /> Reading your garment...</> : "Analyze garment"}
          </button>
          {file && !loading && <button className="secondary" onClick={() => { setFile(null); setPreview(""); setResult(null); }}>Clear</button>}
          {error && <p className="error">{error}</p>}
        </div>
        {result && <Results result={result} />}
      </section>
    </main>
  );
}

function Results({ result }) {
  return (
    <section className="results">
      <div className="result-heading">
        <div><p className="eyebrow">ANALYSIS RESULT</p><h2>{result.category}</h2></div>
        <span className="confidence">{Math.round(result.confidence * 100)}% confidence · {result.model}</span>
      </div>
      <div className="color-card"><span className="swatch" style={{ backgroundColor: result.color.hex }} /><div><small>DOMINANT COLOR</small><strong>{result.color.name}</strong><code>{result.color.hex}</code></div></div>
      {result.grabcut_image && <div className="segmentation-card">
        <div><p className="eyebrow">SEGMENTATION PREVIEW</p><p className="segmentation-copy">GrabCut isolated the garment pixels used for color analysis.</p></div>
        <img src={result.grabcut_image} alt="GrabCut garment segmentation" />
      </div>}
      <h3>Visual inspiration</h3>
      <div className="recommendation-gallery">{result.recommendations.map((item, index) => <figure key={`${item.item}-${item.color}`}>
        <div className="recommendation-number">0{index + 1}</div>
        {item.image_url ? <img className="recommendation-image" src={assetUrl(item.image_url)} alt={`${item.category} inspiration`} /> : <div className="image-placeholder">IMAGE<br />UNAVAILABLE</div>}
        <figcaption>{item.category} reference</figcaption>
      </figure>)}</div>
      <div className="text-recommendations">
        <h3>{result.category === "top" ? "Compatible bottoms" : result.category === "bottom" ? "Compatible tops" : "Fullwear styling"}</h3>
        <ol>{result.recommendations.map((item) => <li key={`${item.item}-${item.reason}`}>
          <div><strong>{item.item}</strong><small>{item.color}</small></div>
          <p>{item.reason}</p>
        </li>)}</ol>
      </div>
    </section>
  );
}

createRoot(document.getElementById("root")).render(<StrictMode><App /></StrictMode>);
