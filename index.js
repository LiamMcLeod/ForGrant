// Network is called Jarvis
const Koa = require('koa');
// const Normalizer = require('neural-data-normalizer/dist/src/normalizer.js');
// var Normalizer = require("./normalizer");
const Normalizer = require('neural-data-normalizer/dist/src/normalizer.js');

const app = new Koa();

var path = require('path');
var fs = require('fs');

var env = require('dotenv').config();

var synaptic = require('synaptic');
var Neuron = synaptic.Neuron,
    Layer = synaptic.Layer,
    Network = synaptic.Network,
    Trainer = synaptic.Trainer,
    Architect = synaptic.Architect;

// Data Normalise
var data = [
    {
        input: [0, 0, 0],
        output: [false]
    },
    {
        input: [0, 0, 1],
        output: [true]
    },
    {
        input: [0, 1, 0],
        output: [true]
    },
    {
        input: [0, 1, 1],
        output: [false]
    },
    {
        input: [1, 0, 0],
        output: [true]
    },
    {
        input: [1, 0, 1],
        output: [false]
    },
    {
        input: [1, 1, 1],
        output: [true]
    }
];


const normaliser = new Normalizer.Normalizer(data);
normaliser.setOutputProperties(['output']);
normaliser.normalize();

const nbrInputs = normaliser.getInputLength();
// console.log(nbrInputs);
const nbrOutputs = normaliser.getOutputLength();
// console.log(nbrOutputs);

const metadata = normaliser.getDatasetMetaData();
// console.log(metadata);
const inputs = normaliser.getBinaryInputDataset();
// console.log(inputs);
const outputs = normaliser.getBinaryOutputDataset();
// console.log(outputs);

// console.log(normaliser.getBinaryInputDataset()[0]);
// console.log(normaliser.getBinaryInputDataset()[1]);


var trainingSet = [];

/** Start Normalised Input **/
// for (var i in inputs) {
//     trainingSet.push({
//         input: inputs[i]
//         output: outputs[i]//[0]
//     });
//     console.log(trainingSet[i]);
// }
/** End Normalised Input **/

/** Start Raw Input **/
for (var i in inputs) {
    trainingSet.push({
        input: data[i].input,
        output: outputs[i]//[0]
    });
    console.log(trainingSet[i]);
}
/** End Raw Input **/

var jarvisOut = [];

/** perceptron(inputs, layers/hidden, outputs)**/
var network = new Architect.Perceptron(3, 3, 1);
var trainer = new Trainer(network);
/** train(set, options)**/
trainer.train(trainingSet, {
    log: 1,
    rate: .1                            // Error Rate
    /**  Self-adjusting Rate
     /**|| function (it, err) {
        if (it % 1000 === 0) {
            lastErrs.unshift(err);
            if (lastErrs.length >= 100) {
                lastErrs.length = 100;
            }
            if (lastErrs.length > 5 && currRate > 0.002 && lastErrs[0] > lastErrs[1] + lastErrs[2] + lastErrs[3] / 3) {
                currRate = currRate * 0.985;
            }
            if (lastErrs.length > 50 && currRate > 0.002 && lastErrs[50] < lastErrs[0]) {
                currRate = currRate * 0.995;
            }
        }
        return currRate
    }**/,
    iterations: 1500000,                    //1500000,
    shuffle: true,
    cost: Trainer.cost.CROSS_ENTROPY        //Trainer.cost.CROSS_ENTROPY //Trainer.cost.MSE
});

/** Value between 0 and 1 **/
var results = [];
results.push(network.activate([0, 0, 0]));
results.push(network.activate([0, 0, 1]));
results.push(network.activate([0, 1, 0]));
results.push(network.activate([0, 1, 1]));
results.push(network.activate([1, 0, 0]));
results.push(network.activate([1, 0, 1]));
results.push(network.activate([1, 1, 1]));
console.log(results);

jarvisOut = results;
for (i = 0; i < results.length; i++) {
    if (i === 0) {
        jarvisOut[i] = "000|" + invert(Math.round(results[i])) + " (" + results[i] + ")";
    }
    if (i === 1) {
        jarvisOut[i] = "001|" + invert(Math.round(results[i])) + " (" + results[i] + ")";
    }
    if (i === 2) {
        jarvisOut[i] = "010|" + invert(Math.round(results[i])) + " (" + results[i] + ")";
    }
    if (i === 3) {
        jarvisOut[i] = "011|" + invert(Math.round(results[i])) + " (" + results[i] + ")";
    }
    if (i === 4) {
        jarvisOut[i] = "100|" + invert(Math.round(results[i])) + " (" + results[i] + ")";
    }
    if (i === 5) {
        jarvisOut[i] = "101|" + invert(Math.round(results[i])) + " (" + results[i] + ")";
    }
    if (i === 6) {
        jarvisOut[i] = "111|" + invert(Math.round(results[i])) + " (" + results[i] + ")";
    }
}
/** For odd or even parity **/
function invert(i){if(i===0){return 1}else{return 0}}

/** XOR Start **/
// var network = new Architect.Perceptron(2, 3, 1);
// var trainer = new Trainer(network);
// trainer.XOR({
//     iterations: 100000,
//     error: .0001,
//     rate: 1
// });
// var jarvisOut =[];
// jarvisOut.push(network.activate([0,0]));
// jarvisOut.push(network.activate([0,1]));
// jarvisOut.push(network.activate([1,0]));
// jarvisOut.push(network.activate([1,1]));
// jarvisOut.push(network);
// jarvisOut.push(trainer);
// console.log(jarvisOut);
/** XOR End **/

var contentBody = 'Hello World';
contentBody = '<h1> Data </h1>';
contentBody += '<hr><h2> Raw Data </h2><pre>' + JSON.stringify(data, null, 2) + '</pre>';
contentBody += '<h1> Normalisation </h1>';
contentBody += '<hr> <h2> Training Set: </h2>' + JSON.stringify(trainingSet, null, 2) + '</pre>';
contentBody += '<hr> <h1> Neural Network </h1>';
contentBody += '<hr> <h2> Output: </h2><pre>' + JSON.stringify(jarvisOut, null, 2) + '</pre>';
contentBody += '<hr>';

// @formatter:off
app.use(async ctx => {
    ctx.body = contentBody;
});
app.listen(3000);
// @formatter:on

exports = module.exports;