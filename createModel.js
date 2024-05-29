const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");

// Load and preprocess the MNIST dataset
async function loadData() {
 const mnist = require("mnist");
 const set = mnist.set(60000, 10000);

 const train = set.training;
 const test = set.test;

 const trainX = tf.tensor2d(
  train.map((d) => d.input),
  [train.length, 28 * 28]
 );
 const trainY = tf.tensor2d(
  train.map((d) => [d.output[0], d.output[1]]),
  [train.length, 2]
 );

 const testX = tf.tensor2d(
  test.map((d) => d.input),
  [test.length, 28 * 28]
 );
 const testY = tf.tensor2d(
  test.map((d) => [d.output[0], d.output[1]]),
  [test.length, 2]
 );

 return { trainX, trainY, testX, testY };
}

// Build the model
function createModel() {
 const model = tf.sequential();

 model.add(tf.layers.dense({ inputShape: [28 * 28], units: 128, activation: "relu" }));
 model.add(tf.layers.dense({ units: 2, activation: "softmax" }));

 model.compile({
  optimizer: tf.train.adam(),
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
 });

 return model;
}

// Train the model
async function trainModel() {
 const { trainX, trainY, testX, testY } = await loadData();
 const model = createModel();

 await model.fit(trainX, trainY, {
  epochs: 5,
  validationData: [testX, testY],
  callbacks: tf.callbacks.earlyStopping({ monitor: "val_loss" }),
 });

 await model.save("file://./model");

 console.log("Model training complete and saved as model.json.");
}

// Execute the training
trainModel().catch(console.error);
