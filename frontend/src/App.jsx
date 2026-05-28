import { useMemo, useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const influencerOptions = ["Mega", "Macro", "Micro", "Nano"];

function App() {
    const [form, setForm] = useState({
        tv: 40,
        radio: 18,
        social_media: 4,
        influencer: "Macro",
    });
    const [prediction, setPrediction] = useState(null);
    const [status, setStatus] = useState("idle");
    const [error, setError] = useState("");

    const budgetTotal = useMemo(() => {
        return Number(form.tv) + Number(form.radio) + Number(form.social_media);
    }, [form]);

    const handleChange = (event) => {
        const { name, value } = event.target;
        setForm((prev) => ({ ...prev, [name]: value }));
    };

    const handlePredict = async () => {
        setStatus("loading");
        setError("");
        try {
            const response = await fetch(`${API_BASE}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    tv: Number(form.tv),
                    radio: Number(form.radio),
                    social_media: Number(form.social_media),
                    influencer: form.influencer,
                }),
            });

            if (!response.ok) {
                const detail = await response.json();
                throw new Error(detail?.detail || "Prediction failed");
            }

            const data = await response.json();
            setPrediction(data.predicted_sales);
            setStatus("success");
        } catch (err) {
            setError(err.message);
            setStatus("error");
        }
    };

    return (
        <div className="app">
            <header className="header">
                <p className="eyebrow">Prediction rapide</p>
                <h1>Predire les ventes d’une campagne</h1>
                <p className="lead">
                    Renseignez les budgets et obtenez une prediction
                    instantanee.
                </p>
            </header>

            <section className="content">
                <div className="card">
                    <h2>Parametres</h2>
                    <div className="form">
                        <label>
                            TV (M)
                            <input
                                name="tv"
                                type="number"
                                min="0"
                                value={form.tv}
                                onChange={handleChange}
                            />
                        </label>
                        <label>
                            Radio (M)
                            <input
                                name="radio"
                                type="number"
                                min="0"
                                value={form.radio}
                                onChange={handleChange}
                            />
                        </label>
                        <label>
                            Social Media (M)
                            <input
                                name="social_media"
                                type="number"
                                min="0"
                                value={form.social_media}
                                onChange={handleChange}
                            />
                        </label>
                        <label>
                            Influencer
                            <select
                                name="influencer"
                                value={form.influencer}
                                onChange={handleChange}
                            >
                                {influencerOptions.map((option) => (
                                    <option key={option} value={option}>
                                        {option}
                                    </option>
                                ))}
                            </select>
                        </label>
                    </div>
                    <button
                        className="primary block"
                        type="button"
                        onClick={handlePredict}
                    >
                        Predire les ventes
                    </button>
                </div>

                <div className="card result">
                    <p className="panel-label">Prediction</p>
                    <h2 className="panel-value">
                        {prediction !== null
                            ? `${prediction.toFixed(2)}M`
                            : "--"}
                    </h2>
                    <p className="panel-helper">
                        Budget total: {budgetTotal.toFixed(1)}M
                    </p>
                    <div className="panel-status">
                        <span className={`pill ${status}`}>{status}</span>
                        {error && <p className="error">{error}</p>}
                    </div>
                </div>
            </section>
        </div>
    );
}

export default App;
