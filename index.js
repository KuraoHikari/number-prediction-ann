const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const bodyParser = require("body-parser");

const app = express();
const upload = multer({ dest: "uploads/" });

// Load the model
const modelPath = "file://./model/model.json";
let model;
async function loadModel() {
 model = await tf.loadLayersModel(modelPath);
}
loadModel();

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Endpoint to handle image uploads and predictions
app.post("/predict", upload.single("digit"), async (req, res) => {
 if (!req.file) {
  return res.status(400).send("No image uploaded.");
 }

 try {
  // Preprocess the image
  const image = fs.readFileSync(req.file.path);
  const tensor = tf.node
   .decodeImage(image, 1) // Decode image as grayscale
   .resizeNearestNeighbor([28, 28]) // Resize to 28x28 pixels
   .toFloat()
   .div(tf.scalar(255.0)) // Normalize pixel values to [0, 1]
   .reshape([1, 28 * 28]); // Flatten the image to [1, 28*28]

  // Predict the image
  const predictions = await model.predict(tensor).data();
  console.log("🚀 ~ app.post ~ predictions:", predictions);
  console.log("🚀 ~ app.post ~ predictions[0]:", predictions[0].toFixed(2), predictions[1].toFixed(2));
  let prediction = predictions[0].toFixed(2) > predictions[1].toFixed(2) ? "It's a 0" : "It's a 1";

  // Send the result
  res.status(200).json({ prediction, predictions });
 } catch (error) {
  res.status(500).send(error.toString());
 }

 // Clean up
 fs.unlinkSync(req.file.path);
});

// Start the server
const port = 3000;
app.listen(port, () => {
 console.log(`Server running on http://localhost:${port}`);
});
