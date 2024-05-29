const express = require("express");
const tf = require("@tensorflow/tfjs-node");
const { createCanvas } = require("canvas");
const app = express();
const port = 3000;

// Load the model
async function loadModel() {
 const model = await tf.loadLayersModel("file://./model/model.json");
 return model;
}

// Generate an image with a number (0 or 1)
function generateImage(number) {
 const canvas = createCanvas(28, 28);
 const ctx = canvas.getContext("2d");

 ctx.fillStyle = "white";
 ctx.fillRect(0, 0, 28, 28);
 ctx.fillStyle = "black";
 ctx.font = "24px Arial";
 ctx.fillText(number.toString(), 5, 24);

 return canvas.toBuffer("image/png");
}

// Convert the image to tensor
function imageToTensor(imageBuffer) {
 const image = tf.node.decodeImage(imageBuffer, 1);
 const resizedImage = tf.image.resizeBilinear(image, [28, 28]);
 const tensor = resizedImage.expandDims(0).toFloat().div(tf.scalar(255.0));
 return tensor.reshape([1, 28 * 28]);
}

let model;

// Middleware to load the model
app.use(async (req, res, next) => {
 if (!model) {
  model = await loadModel();
 }
 next();
});

// Endpoint to generate an image
app.get("/generate-image/:number", (req, res) => {
 const number = parseInt(req.params.number, 10);
 if (number !== 0 && number !== 1) {
  return res.status(400).send("Only 0 or 1 is allowed.");
 }

 const imageBuffer = generateImage(number);
 res.type("png");
 res.send(imageBuffer);
});

// Endpoint to predict the number from the generated image
app.post("/predict-number", async (req, res) => {
 const number = req.query.number;
 if (number !== "0" && number !== "1") {
  return res.status(400).send("Only 0 or 1 is allowed.");
 }

 const imageBuffer = generateImage(number);
 const tensor = imageToTensor(imageBuffer);
 const prediction = model.predict(tensor);
 const predictedClass = prediction.argMax(1).dataSync()[0];

 res.send({ predicted: predictedClass, prediction });
});

app.listen(port, () => {
 console.log(`Server is running on http://localhost:${port}`);
});
