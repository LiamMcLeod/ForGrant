// Network is called Jarvis
const Koa = require('koa');
// const Normalizer = require('neural-data-normalizer/dist/src/normalizer.js');
// var Normalizer = require("./normalizer");
const Normalizer = require('neural-data-normalizer/dist/src/normalizer.js');

const app = new Koa();

var path = require('path');
var fs = require('fs');

var env = require('dotenv').config();

//
// var mysql      = require('mysql');
// var connection = mysql.createConnection({
//     port     : '33060',
//     host     : 'localhost',
//     user     : 'homestead',
//     password : 'secret',
//     database : 'NeuralNetDB',
//     insecureAuth: true
// });
// connection.connect(function(err) {
//     if (err) {
//         console.error('error connecting: ' + err.stack);
//         return;
//     }
//
//     console.log('connected as id ' + connection.threadId);
// });
// console.log(connection);

var synaptic = require('synaptic');
var Neuron = synaptic.Neuron,
    Layer = synaptic.Layer,
    Network = synaptic.Network,
    Trainer = synaptic.Trainer,
    Architect = synaptic.Architect;

// Data Normalise
// var data = [
//     {
//         input: [0, 0, 0],
//         output: [false]
//     },
//     {
//         input: [0, 0, 1],
//         output: [true]
//     },
//     {
//         input: [0, 1, 0],
//         output: [true]
//     },
//     {
//         input: [0, 1, 1],
//         output: [false]
//     },
//     {
//         input: [1, 0, 0],
//         output: [true]
//     },
//     {
//         input: [1, 0, 1],
//         output: [false]
//     },
//     {
//         input: [1, 1, 1],
//         output: [true]
//     }
// ];
// var unData = [
//     {input: [725, 50], output: [0]},
//     {input: [746, 53], output: [0]},
//     {input: [1084, 76], output: [0]},
//     {input: [572, 39], output: [0]},
//     {input: [735, 52], output: [1]},
//     {input: [539, 52], output: [1]},
//     {input: [1212, 77], output: [0]},
//     {input: [676, 59], output: [0]},
//     {input: [806, 46], output: [1]},
//     {input: [1102, 62], output: [1]},
//     {input: [762, 74], output: [1]},
//     {input: [1292, 92], output: [0]},
//     {input: [792, 58], output: [0]},
//     {input: [966, 52], output: [1]},
//     {input: [664, 41], output: [0]}
// ];
var data = [
    {
        input: [0.39960822722820766, 0.00026429056860227673],
        output: [0]
    }, {
        input: [0.3550440744368266, 0.015474842054922056],
        output: [0]
    },
    {
        input: [0.3653281096963761, 0.007631704800020123],
        output: [0]
    }, {

        input: [0.5308521057786484, 0.0006720531601601687],
        output: [0]
    },
    {
        input: [0.2801175318315377, 0.004487905560170147],
        output: [0]
    },
    {
        input: [0.3599412340842311, 0.012590299277605799],
        output: [1]
    },
    {
        input: [0.2639569049951028, 0.025376928691887513],
        output: [1]
    },
    {
        input: [0.5935357492654261, 0.0001887789775730786],
        output: [0]
    },
    {
        input: [0.33104799216454456, 0.023912003825920625],
        output: [0]
    },
    {
        input: [0.39471106758080315, 0.0004656548113468606],
        output: [1]
    },
    {
        input: [0.5396669931439765, 0.00023408593219059748],
        output: [1]
    },
    {
        input: [0.3731635651322233, 0.04607213873996324],
        output: [1]
    },
    {
        input: [0.6327130264446621, 0.00034735331873447794],
        output: [0]
    },
    {
        input: [0.3878550440744368, 0.010649651388154713],
        output: [0]
    },
    {
        input: [0.4730656219392752, 0.00012333559868105137],
        output: [1]
    },
    {
        input: [0.32517140058765914, 0.0009640313121397348],
        output: [0]
    },
    {
        input: [0.6620959843290891, 0.03474791713861414],
        output: [1]
    },
    {
        input: [0.41234084231145934, 0.010143723728259002],
        output: [1]
    },
    {
        input: [0.579333986287953, 0.0005159958720329927],
        output: [0]
    },
    {
        input: [0.5205680705190989, 0.00007047748496058492],
        output: [1]
    },
    {
        input: [0.5822722820763957, 0.001341589267285892],
        output: [0]
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

function normalised(data, bool) {
    if (typeof bool === 'undefined') {
        bool = false
    }
    var i=0;
    if (bool === true) {
        for (i in inputs) {
            trainingSet.push({
                input: data[i].input,
                output: outputs[i]//[0]
            });
            // console.log(trainingSet[i]);
        }
    } else {
        for (i in inputs) {
            trainingSet.push({
                input: inputs[i],
                output: outputs[i]//[0]
            });
            //console.log(trainingSet[i]);
        }
    }
}

/** Start Normalised Input **/
// for (var i in inputs) {
//     trainingSet.push({
//         input: inputs[i],
//         output: outputs[i]//[0]
//     });
//     //console.log(trainingSet[i]);
// }
/** End Normalised Input **/

/** Start Raw Input **/
// for (var i in inputs) {
//     trainingSet.push({
//         input: data[i].input,
//         output: outputs[i]//[0]
//     });
//     // console.log(trainingSet[i]);
// }
/** End Raw Input **/

normalised(data, true);

var jarvisOut = [];

/** perceptron(inputs, layers/hidden, outputs)**/
// var network = new Architect.Perceptron(3, 3, 1);
var network = new Architect.Perceptron(2, 3, 1);
var trainer = new Trainer(network);
/** train(set, options)**/
var test = trainer.train(trainingSet, {
    //error: .003,                           //Error Rate
    log: 1,
    rate: .001,                               //Learn rate
    iterations: 5000000, //0,                    //1500000,
    shuffle: true,
    cost: Trainer.cost.CROSS_ENTROPY        //Trainer.cost.CROSS_ENTROPY //Trainer.cost.MSE
});

/** Value between 0 and 1 **/
var results = [];
// results.push(network.activate([0, 0, 0]));
// results.push(network.activate([0, 0, 1]));
// results.push(network.activate([0, 1, 0]));
// results.push(network.activate([0, 1, 1]));
// results.push(network.activate([1, 0, 0]));
// results.push(network.activate([1, 0, 1]));
// results.push(network.activate([1, 1, 1]));
results.push(network.activate([0.5935357492654261, 0.0001887789775730786])); //0
results.push(network.activate([0.5205680705190989, 0.00007047748496058492])); //1
console.log(results);

jarvisOut = results;
for (i = 0; i < results.length; i++) {
    jarvisOut[i] = results[i];
    // if (i === 0) {
    //     jarvisOut[i] = "000|" + invert(Math.round(results[i])) + " (" + results[i] + ")";
    // }
    // if (i === 1) {
    //     jarvisOut[i] = "001|" + invert(Math.round(results[i])) + " (" + results[i] + ")";
    // }
    // if (i === 2) {
    //     jarvisOut[i] = "010|" + invert(Math.round(results[i])) + " (" + results[i] + ")";
    // }
    // if (i === 3) {
    //     jarvisOut[i] = "011|" + invert(Math.round(results[i])) + " (" + results[i] + ")";
    // }
    // if (i === 4) {
    //     jarvisOut[i] = "100|" + invert(Math.round(results[i])) + " (" + results[i] + ")";
    // }
    // if (i === 5) {
    //     jarvisOut[i] = "101|" + invert(Math.round(results[i])) + " (" + results[i] + ")";
    // }
    // if (i === 6) {
    //     jarvisOut[i] = "111|" + invert(Math.round(results[i])) + " (" + results[i] + ")";
    // }
}

/** For odd or even parity **/
// @formatter:off
function invert(n){return 0===n?1:0}
// @formatter:on

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