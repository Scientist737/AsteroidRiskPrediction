function predictRisk() {
    let diameter = document.getElementById("diameter").value;
    let velocity = document.getElementById("velocity").value;
    let distance = document.getElementById("distance").value;

    if (diameter === "" || velocity === "" || distance === "") {
        document.getElementById("result").innerText = "Please enter all values.";
        return;
    }

    let data = {
        "diameter": parseFloat(diameter),
        "velocity": parseFloat(velocity),
        "distance": parseFloat(distance)
    };

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = "Prediction: " + data.prediction;
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("result").innerText = "Error predicting risk.";
    });
}
