async function analyzeSentiment() {
    const text = document.getElementById("textInput").value;
    const resultDiv = document.getElementById("result");

    if (!text) {
        resultDiv.innerHTML = "Please enter some text.";
        return;
    }

    resultDiv.innerHTML = "Analyzing...";

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        resultDiv.innerHTML = `
            <strong>Sentiment:</strong> ${data.sentiment}<br>
            <strong>Confidence:</strong> ${data.confidence}
        `;
    } catch (error) {
        resultDiv.innerHTML = "Error connecting to backend.";
    }
}
