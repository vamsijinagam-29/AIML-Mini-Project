import { useEffect, useMemo, useState } from 'react'
import { Activity, ArrowUpRight, BarChart3, CarFront, Check, ChevronDown, CircleAlert, Database, Gauge, History, LoaderCircle, RotateCcw, ShieldCheck, Sparkles, Zap } from 'lucide-react'

type Metadata = {
  features: string[]
  target: string
  dataset: {
    rows_before_cleaning: number
    rows_after_cleaning: number
    duplicate_rows_removed: number
    invalid_rows_removed: number
    price_min: number
    price_max: number
    price_median: number
    price_mean: number
    year_min: number
    year_max: number
    source_filename: string
  }
  numeric_ranges: {
    year: { min: number; max: number }
    km_driven: { min: number; max: number }
  }
  options: Record<string, string[]>
  models_by_brand: Record<string, string[]>
  model: {
    selected_model: string
    selection_metric: string
    selection_rule: string
    test_rows: number
    selected_metrics: { model: string; mae: number; rmse: number; r2: number }
    cross_validation: { mae_mean: number; mae_std: number; rmse_mean: number; r2_mean: number }
    comparisons: Array<{ model: string; mae: number; rmse: number; r2: number }>
  }
}

type FormState = {
  name: string
  year: string
  km_driven: string
  fuel: string
  seller_type: string
  transmission: string
  owner: string
}

type Prediction = {
  estimated_price: number
  estimated_price_inr: string
  model: string
  input: FormState
  createdAt: string
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')
const HISTORY_KEY = 'carwise-ai-history'

const formatINR = (value: number) => new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(value)
const formatCompact = (value: number) => new Intl.NumberFormat('en-IN', { notation: 'compact', maximumFractionDigits: 1 }).format(value)

function firstOption(options: Record<string, string[]>, key: string, preferred?: string) {
  return options[key]?.find((option) => option === preferred) ?? options[key]?.[0] ?? ''
}

function firstBrand(data: Metadata) {
  return data.options.brand?.[0] ?? ''
}

function brandForModel(data: Metadata, modelName: string) {
  return Object.entries(data.models_by_brand).find(([, models]) => models.includes(modelName))?.[0] ?? firstBrand(data)
}

function buildDefaultForm(data: Metadata): FormState {
  const brand = firstBrand(data)
  return {
    name: data.models_by_brand[brand]?.[0] ?? data.options.name?.[0] ?? '',
    year: String(data.numeric_ranges.year.max),
    km_driven: String(Math.round(data.dataset.price_median > 0 ? 60000 : 50000)),
    fuel: firstOption(data.options, 'fuel', 'Petrol'),
    seller_type: firstOption(data.options, 'seller_type', 'Individual'),
    transmission: firstOption(data.options, 'transmission', 'Manual'),
    owner: firstOption(data.options, 'owner', 'First Owner'),
  }
}

function App() {
  const [metadata, setMetadata] = useState<Metadata | null>(null)
  const [form, setForm] = useState<FormState | null>(null)
  const [brand, setBrand] = useState('')
  const [result, setResult] = useState<Prediction | null>(null)
  const [history, setHistory] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(false)
  const [metadataError, setMetadataError] = useState('')
  const [predictionError, setPredictionError] = useState('')

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(HISTORY_KEY) ?? '[]') as Prediction[]
      if (Array.isArray(saved)) setHistory(saved)
    } catch {
      setHistory([])
    }

    fetch(`${API_BASE}/metadata`)
      .then(async (response) => {
        if (!response.ok) throw new Error((await response.json()).detail ?? 'Metadata request failed')
        return response.json() as Promise<Metadata>
      })
      .then((data) => {
        setMetadata(data)
        setBrand(firstBrand(data))
        setForm(buildDefaultForm(data))
      })
      .catch((error: Error) => setMetadataError(error.message || 'The prediction service is unavailable.'))
  }, [])

  const maxMae = useMemo(() => Math.max(...(metadata?.model.comparisons.map((item) => item.mae) ?? [1])), [metadata])
  const modelShortName = metadata?.model.selected_model.replace(' Regressor', '').replace(' Regression', '') ?? '—'
  const modelOptions = metadata && brand ? metadata.models_by_brand[brand] ?? [] : []

  function updateField(field: keyof FormState, value: string) {
    setForm((current) => current ? { ...current, [field]: value } : current)
  }

  async function submitPrediction(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!form || !metadata) return
    setLoading(true)
    setPredictionError('')
    setResult(null)
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, year: Number(form.year), km_driven: Number(form.km_driven) }),
      })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail ?? 'Prediction request failed')
      const next: Prediction = { ...body, input: form, createdAt: new Date().toISOString() }
      setResult(next)
      const nextHistory = [next, ...history].slice(0, 6)
      setHistory(nextHistory)
      localStorage.setItem(HISTORY_KEY, JSON.stringify(nextHistory))
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : 'Prediction failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  function loadHistory(item: Prediction) {
    if (metadata) setBrand(brandForModel(metadata, item.input.name))
    setForm(item.input)
    setResult(item)
    document.getElementById('estimate')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  function resetForm() {
    if (metadata) {
      setBrand(firstBrand(metadata))
      setForm(buildDefaultForm(metadata))
      setResult(null)
      setPredictionError('')
    }
  }

  return (
    <div className="app-shell">
      <aside className="side-rail">
        <div className="brand-lockup">
          <div className="brand-mark"><span></span><span></span><span></span></div>
          <div><strong>CARWISE <i>AI</i></strong><small>RESALE INTELLIGENCE</small></div>
        </div>
        <nav className="rail-nav" aria-label="Primary navigation">
          <a className="active" href="#estimate"><Gauge size={17} /> Estimate <span>01</span></a>
          <a href="#evidence"><BarChart3 size={17} /> Model evidence</a>
          <a href="#model"><Sparkles size={17} /> ML model used</a>
          <a href="#history"><History size={17} /> Local history</a>
        </nav>
        <div className="rail-footer">
          <div className="rail-status"><span className="status-dot" /> API LINKED</div>
          <p>Built from the supplied Car Dekho dataset. No synthetic fallback.</p>
          <div className="version-row"><span>ENGINE</span><b>v1.0 / LIVE</b></div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="crumb"><span>WORKSPACE</span><b>/</b> PRICE ESTIMATE</div>
          <div className="topbar-right"><span className="live-badge"><span className="status-dot" /> MODEL ONLINE</span><span className="topbar-date">INDIA / INR</span></div>
        </header>

        <section className="hero-block">
          <div className="hero-copy">
            <div className="eyebrow"><span className="signal-line"></span> DATA-BACKED VEHICLE VALUATION</div>
            <h1>Price the <em>next move.</em></h1>
            <p>Estimate an Indian used car’s resale value with a model trained on real market listings — transparent inputs, observable evidence.</p>
            <div className="hero-pills"><span><Check size={13} /> REAL DATASET</span><span><Check size={13} /> LIVE API</span><span><Check size={13} /> INR OUTPUT</span></div>
          </div>
          <div className="hero-instrument" aria-label="Model status instrument">
            <div className="instrument-ring"><div className="ring-core"><Sparkles size={18} /><b>{metadata ? modelShortName.toUpperCase() : 'LOADING'}</b><small>SELECTED MODEL</small></div></div>
            <div className="instrument-readout"><span>FIT SCORE</span><b>{metadata ? `${(metadata.model.selected_metrics.r2 * 100).toFixed(1)}%` : '—'}</b><small>TEST R²</small></div>
          </div>
        </section>

        {metadataError && <div className="service-alert"><CircleAlert size={17} /><div><b>Prediction service unavailable</b><span>{metadataError}. Start the FastAPI service and refresh; this interface does not use mock prices.</span></div></div>}

        <section className="workbench" id="estimate">
          <div className="section-label"><span>01</span><div><b>ESTIMATE VEHICLE</b><small>Use only details present in the training data</small></div></div>
          <div className="workbench-layout">
            <form className="form-card" onSubmit={submitPrediction}>
              <div className="form-header"><div><h2>Vehicle profile</h2><p>Set the known attributes. The model will do the rest.</p></div><button className="ghost-button" type="button" onClick={resetForm}><RotateCcw size={14} /> Reset</button></div>
              <div className="form-grid">
                <label className="field"><span>Vehicle brand <b>*</b></span><div className="input-wrap select-wrap"><CarFront size={16} /><select value={brand} onChange={(event) => { const nextBrand = event.target.value; setBrand(nextBrand); updateField('name', metadata?.models_by_brand[nextBrand]?.[0] ?? '') }} required><option value="" disabled>Select a brand</option>{metadata?.options.brand?.map((option) => <option key={option} value={option}>{option}</option>)}</select><ChevronDown size={15} /></div><small>Derived from the source model names</small></label>
                <label className="field"><span>Vehicle model <b>*</b></span><div className="input-wrap select-wrap model-select-wrap"><CarFront size={16} /><select value={form?.name ?? ''} onChange={(event) => updateField('name', event.target.value)} required><option value="" disabled>Select a vehicle model</option>{modelOptions.map((name) => <option key={name} value={name}>{name}</option>)}</select><ChevronDown size={15} /></div><small>{modelOptions.length.toLocaleString('en-IN')} models available for {brand || 'selected brand'}</small></label>
                <label className="field"><span>Manufacturing year <b>*</b></span><div className="input-wrap"><input type="number" min={metadata?.numeric_ranges.year.min} max={metadata?.numeric_ranges.year.max} value={form?.year ?? ''} onChange={(event) => updateField('year', event.target.value)} required /><span className="suffix">YEAR</span></div><small>{metadata?.numeric_ranges.year.min ?? '—'}–{metadata?.numeric_ranges.year.max ?? '—'} supported</small></label>
                <label className="field"><span>Distance driven <b>*</b></span><div className="input-wrap"><input type="number" min={metadata?.numeric_ranges.km_driven.min} max={metadata?.numeric_ranges.km_driven.max} value={form?.km_driven ?? ''} onChange={(event) => updateField('km_driven', event.target.value)} required /><span className="suffix">KM</span></div><small>Actual odometer reading</small></label>
                {(['fuel', 'seller_type', 'transmission', 'owner'] as const).map((field) => (
                  <label className="field" key={field}><span>{field === 'seller_type' ? 'Seller type' : field === 'owner' ? 'Ownership' : field[0].toUpperCase() + field.slice(1)} <b>*</b></span><div className="input-wrap select-wrap"><select value={form?.[field] ?? ''} onChange={(event) => updateField(field, event.target.value)} required>{metadata?.options[field]?.map((option) => <option key={option} value={option}>{option}</option>)}</select><ChevronDown size={15} /></div><small>Source feature</small></label>
                ))}
              </div>
              {predictionError && <div className="inline-error"><CircleAlert size={15} /> {predictionError}</div>}
              <div className="form-actions"><button className="primary-button" type="submit" disabled={loading || !metadata || !form}>{loading ? <><LoaderCircle className="spin" size={17} /> RUNNING MODEL</> : <><Zap size={17} /> ESTIMATE RESALE PRICE <ArrowUpRight size={17} /></>}</button><span className="form-note"><ShieldCheck size={14} /> Validated against source ranges</span></div>
            </form>

            <aside className={`result-card ${result ? 'has-result' : ''}`} aria-live="polite">
              <div className="result-top"><span className="result-kicker">ESTIMATED RESALE VALUE</span><span className="result-icon"><Activity size={16} /></span></div>
              <div className="price-display">{loading ? <span className="price-skeleton"></span> : result ? result.estimated_price_inr : '₹ —'}</div>
              <p className="result-caption">{result ? `Calculated by ${result.model}` : 'Complete the profile to generate a model estimate.'}</p>
              <div className="result-divider"></div>
              <div className="result-meta"><span>OUTPUT CURRENCY <b>INR / ₹</b></span><span>CONFIDENCE CONTEXT <b>TEST R² {metadata ? `${(metadata.model.selected_metrics.r2 * 100).toFixed(1)}%` : '—'}</b></span></div>
              <div className="result-footnote"><CircleAlert size={14} /> Estimate only — condition, location and timing can change the final sale price.</div>
            </aside>
          </div>
        </section>

        <section className="evidence-section" id="evidence">
          <div className="section-label"><span>02</span><div><b>MODEL EVIDENCE</b><small>Actual evaluation results from the held-out test set</small></div></div>
          <div className="evidence-grid">
            <article className="panel dataset-panel"><div className="panel-heading"><div><span className="panel-kicker">DATASET PULSE</span><h3>What the model sees</h3></div><Database size={18} /></div><div className="metric-grid"><div><span>ROWS USED</span><b>{metadata ? metadata.dataset.rows_after_cleaning.toLocaleString('en-IN') : '—'}</b><small>after cleaning</small></div><div><span>MEDIAN PRICE</span><b>{metadata ? formatCompact(metadata.dataset.price_median) : '—'}</b><small>₹ INR</small></div><div><span>YEAR RANGE</span><b>{metadata ? `${metadata.dataset.year_min}–${metadata.dataset.year_max}` : '—'}</b><small>manufacturing year</small></div><div><span>FEATURES</span><b>{metadata?.features.length ?? '—'}</b><small>source columns</small></div></div><div className="source-line"><span className="mini-dot"></span><span>Source preserved: {metadata?.dataset.source_filename ?? 'uploaded CSV'}</span></div></article>
            <article className="panel metrics-panel"><div className="panel-heading"><div><span className="panel-kicker">SELECTED PIPELINE</span><h3>{metadata?.model.selected_model ?? 'Waiting for artifact'}</h3></div><span className="selected-tag">SELECTED</span></div><div className="score-row"><div><span>MAE</span><b>{metadata ? formatINR(metadata.model.selected_metrics.mae) : '—'}</b></div><div><span>RMSE</span><b>{metadata ? formatINR(metadata.model.selected_metrics.rmse) : '—'}</b></div><div><span>R²</span><b>{metadata ? metadata.model.selected_metrics.r2.toFixed(3) : '—'}</b></div></div><div className="cv-note"><ShieldCheck size={15} /><span>3-fold CV MAE <b>{metadata ? formatINR(metadata.model.cross_validation.mae_mean) : '—'}</b> ± {metadata ? formatINR(metadata.model.cross_validation.mae_std) : '—'}</span></div></article>
          </div>
          <article className="panel comparison-panel"><div className="panel-heading"><div><span className="panel-kicker">MODEL COMPARISON</span><h3>Held-out MAE by candidate</h3></div><span className="chart-note"><span className="orange-dot"></span> LOWER IS BETTER</span></div><div className="bars">{metadata?.model.comparisons.map((item) => <div className={`bar-row ${item.model === metadata.model.selected_model ? 'selected' : ''}`} key={item.model}><span className="bar-label">{item.model}</span><div className="bar-track"><div className="bar-fill" style={{ width: `${Math.max(10, (item.mae / maxMae) * 100)}%` }}></div></div><b>{formatCompact(item.mae)}</b></div>) ?? <div className="empty-state">Metrics appear after the trained artifact loads.</div>}</div></article>
        </section>

        <section className="model-section" id="model">
          <div className="section-label"><span>03</span><div><b>ML MODEL USED</b><small>What powers the estimate behind the number</small></div></div>
          <article className="panel model-explainer">
            <div className="model-explainer-head"><div><span className="panel-kicker">PRODUCTION PIPELINE</span><h3>{metadata?.model.selected_model ?? 'Waiting for model artifact'}</h3><p>{metadata?.model.selection_rule ?? 'The selected model and its evidence will appear after the API loads.'}</p></div><div className="model-badge"><Sparkles size={16} /><span>LIVE<br /><b>JOBLIB</b></span></div></div>
            <div className="model-detail-grid"><div><span>ALGORITHM</span><b>{metadata?.model.selected_model ?? '—'}</b><small>Ensemble regression model</small></div><div><span>TRAINING SPLIT</span><b>80 / 20</b><small>{metadata ? `${metadata.model.test_rows.toLocaleString('en-IN')} held-out rows` : '—'}</small></div><div><span>SELECTION METRIC</span><b>{metadata?.model.selection_metric?.toUpperCase() ?? '—'}</b><small>Lowest held-out score wins</small></div><div><span>PREPROCESSING</span><b>ONE-HOT + SCALE</b><small>Numeric + categorical pipeline</small></div></div>
            <div className="feature-strip"><span>INPUT FEATURES</span><div>{metadata?.features.map((feature) => <b key={feature}>{feature}</b>) ?? <b>—</b>}</div></div>
          </article>
        </section>

        <section className="history-section" id="history">
          <div className="section-label"><span>04</span><div><b>LOCAL HISTORY</b><small>Saved in this browser only</small></div></div>
          <div className="history-panel panel">{history.length === 0 ? <div className="history-empty"><History size={19} /><span>Your completed estimates will appear here.</span></div> : <div className="history-list">{history.map((item, index) => <button className="history-item" key={`${item.createdAt}-${index}`} onClick={() => loadHistory(item)}><div className="history-car"><span>{item.input.name}</span><small>{item.input.year} · {Number(item.input.km_driven).toLocaleString('en-IN')} km · {item.input.fuel}</small></div><b>{item.estimated_price_inr}</b><ArrowUpRight size={15} /></button>)}</div>}</div>
        </section>

        <footer className="site-footer"><span>CARWISE AI / INDIAN USED CAR PRICE PREDICTOR</span><span>MODEL ESTIMATES ARE NOT GUARANTEED SALE PRICES</span><span>REAL DATA. REAL PIPELINE.</span></footer>
      </main>
    </div>
  )
}

export default App
