import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";

// Add a listner to the "Run" button
const runButton = document.getElementById("runButton");
runButton.addEventListener("click", convertImage);

const resetButton = document.getElementById("resetButton");
resetButton.addEventListener("click", resetCanvas);

const PREDICTION_ELEMENT = document.getElementById("prediction");
const INPUT_TEXT = document.getElementById("inputText");
const CANVAS = document.getElementById("canvas");
const CTX = CANVAS.getContext("2d");

const trainingLoss = [];
const validationLoss = [];

// Grab a reference to the MNIST input values (pixel data).
const INPUTS = TRAINING_DATA.inputs;

// Grab referennce to the MNIST output values.
const OUTPUTS = TRAINING_DATA.outputs;

// Shuffle the two arrays in the same way so inputs still match outputs indexes.
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// Input feature Array is 2 dimensional.
const INPUTS_TENSOR = tf.tensor2d(INPUTS);

// Output feature Array is 1 dimensional.
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

//Now actually create and define model architecture
const model = tf.sequential();

model.add(
  tf.layers.dense({ inputShape: [784], units: 32, activation: "relu" })
);
model.add(tf.layers.dense({ units: 16, activation: "relu" }));
model.add(tf.layers.dense({ units: 10, activation: "softmax" })); //softmax ensures that all outputs add up to 1

model.summary();

train();

async function train() {
  //Compile the model with the defined optimiser and specify our loss function to use.
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
    shuffle: true, //Ensure data is shuffled again before using each epoch
    validationSplit: 0.2,
    batchSize: 512, // Update weights after every 512 examples
    epochs: 50, // Go over the data 50 times
    callbacks: { onEpochEnd: logProgress },
  });

  console.log(
    "Average error loss: " +
      Math.sqrt(results.history.loss[results.history.loss.length - 1])
  );
  console.log(
    "Average validation error loss: " +
      Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1])
  );

  OUTPUTS_TENSOR.dispose();
  INPUTS_TENSOR.dispose();
  prepareCanvas();
  drawChart(trainingLoss, validationLoss);
  console.log("Training Loss:", trainingLoss);
  console.log("Validation Loss:", validationLoss);
}

// Stores the initial position of the cursor
let coord = { x: 0, y: 0 };

// This is the flag that we are going to use to trigger drawing
let paint = false;

function prepareCanvas() {
  INPUT_TEXT.innerText =
    "Draw a number between 0 and 9 on the canvas below and then press 'Run'";
  // Fill the entire canvas
  CTX.fillStyle = "black"; // Set fill color to black
  CTX.fillRect(0, 0, CTX.canvas.width, CTX.canvas.height);
  document.addEventListener("mousedown", startPainting);
  document.addEventListener("mouseup", stopPainting);
  document.addEventListener("mousemove", sketch);
}

// Updates the coordianates of the cursor when an event e is triggered to the coordinates where the said event is triggered.
function getPosition(event) {
  const rect = CANVAS.getBoundingClientRect(); // Get the canvas's bounding box
  coord.x = (event.clientX - rect.left) * (CANVAS.width / rect.width);
  coord.y = (event.clientY - rect.top) * (CANVAS.height / rect.height);
}

// The following functions toggle the flag to start and stop drawing
function startPainting(event) {
  paint = true;
  getPosition(event);
}
function stopPainting() {
  paint = false;
}

function sketch(event) {
  if (!paint) return;

  const previousX = coord.x; // Save the last position
  const previousY = coord.y;

  getPosition(event); // Update the current position based on the mouse event

  // Begin the path
  CTX.beginPath();
  CTX.lineWidth = 3;
  CTX.lineCap = "round";
  CTX.strokeStyle = "white";

  // Draw small lines between the previous position and the current position
  CTX.moveTo(previousX, previousY);
  CTX.lineTo(coord.x, coord.y);

  // Render the stroke
  CTX.stroke();
}

function convertImage() {
  const imageData = CTX.getImageData(0, 0, 28, 28); //(starting pos y, start pos x, width (px), height(px))
  console.log("imageData", imageData.data);
  const greyscaleImage = imageData.data.filter(function (value, index) {
    return index % 4 == 0;
  });
  const normalisedGreyscaleImage = greyscaleImage.map((value) => value / 255);
  evaluate(normalisedGreyscaleImage);
  // drawImage(normalisedGreyscaleImage);
}

function resetCanvas() {
  CTX.clearRect(0, 0, CANVAS.width, CANVAS.height);
  CTX.fillRect(0, 0, CTX.canvas.width, CTX.canvas.height);
}

function evaluate(image) {
  let answer = tf.tidy(function () {
    let newInput = tf.tensor1d(image);

    let output = model.predict(newInput.expandDims()); //model.predict expects a batch of values so we need to use expandDims() to turn the 1D array into a 2D arrray
    output.print();
    return output.squeeze().argMax(); // squeeze turns the output tensor from a 2D tensor back to a 1D tensor - counteracts exapndDims. argMax catches the largest value in the tensor
  });

  answer.array().then(function (index) {
    // array() turns the tensor into an array (async process)
    PREDICTION_ELEMENT.innerText = index;
    // the index of the largest answer in the output array is equal to the predicted number
    PREDICTION_ELEMENT.setAttribute("class", "number");
    answer.dispose();
  });
}

function logProgress(epoch, logs) {
  console.log(`Epoch ${epoch + 1}`);
  console.log(`Training Loss: ${logs.loss}`);
  console.log(`Validation Loss: ${logs.val_loss}`);
  // Store the loss values for later plotting
  trainingLoss.push(logs.loss);
  validationLoss.push(logs.val_loss);
  
}

function drawChart(trainingLoss, validationLoss) {
  const CHART = document.getElementById("chart");
  Plotly.newPlot(
    CHART,
    [
      {
        x: [...Array(trainingLoss.length).keys()],
        y: trainingLoss,
        name: "Training Loss"
      },
      {
        x: [...Array(trainingLoss.length).keys()],
        y: validationLoss,
        name: "Validation Loss"
      },
    ],
    {
      xaxis:{
        title:{
          text: "Epochs"
        }
      },
      yaxis:{
        title:{
          text:"Categorical Crossentropy Loss"
        }
      },
      margin: { t: 0 }
    }
  );
}
