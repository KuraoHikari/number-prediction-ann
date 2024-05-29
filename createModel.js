const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const mnist = require("mnist");

// Load and preprocess the MNIST dataset
async function loadData() {
 const set = mnist.set(60000, 10000);

 const train = set.training.filter((d) => d.output[0] === 1 || d.output[1] === 1);
 const test = set.test.filter((d) => d.output[0] === 1 || d.output[1] === 1);

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

 model.add(tf.layers.dense({ inputShape: [28 * 28], units: 256, activation: "relu" }));
 model.add(tf.layers.dropout({ rate: 0.2 }));
 model.add(tf.layers.dense({ units: 128, activation: "relu" }));
 model.add(tf.layers.dropout({ rate: 0.2 }));
 model.add(tf.layers.dense({ units: 64, activation: "relu" }));
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
  epochs: 20,
  validationData: [testX, testY],
  callbacks: tf.callbacks.earlyStopping({ monitor: "val_loss", patience: 20 }),
 });

 await model.save("file://./model");

 console.log("Model training complete and saved as model.json.");
}

// Execute the training
trainModel().catch(console.error);
