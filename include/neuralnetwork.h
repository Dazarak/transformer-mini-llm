#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <iostream>

using namespace std;

class NeuralNetwork {
public:
    std::vector<int> layerSizes;
    std::vector<int> layerActivationF;
    std::vector<float> weights;        // Tous les poids de toutes les couches mis bout à bout
    std::vector<float> biases;         // Tous les biais mis bout à bout
    std::vector<float> neurons;        // Toutes les activations de neurones d'un seul bloc
    
    std::vector<size_t> weightOffsets;
    std::vector<size_t> biasOffsets;
    std::vector<size_t> neuronOffsets;
    
    NeuralNetwork() {}
    NeuralNetwork(const std::vector<int>& sizes, const std::vector<int>& actions) : layerSizes(sizes), layerActivationF(actions) {
        size_t totalWeights = 0;
        size_t totalBiases = 0;
        size_t totalNeurons = 0;

        weightOffsets.push_back(0);
        biasOffsets.push_back(0);
        neuronOffsets.push_back(0);

        for (size_t i = 0; i < sizes.size() - 1; ++i) {
            totalWeights += sizes[i] * sizes[i + 1];
            totalBiases += sizes[i + 1];
            weightOffsets.push_back(totalWeights);
            biasOffsets.push_back(totalBiases);
        }
        for (size_t i = 0; i < sizes.size(); ++i) {
            totalNeurons += sizes[i];
            neuronOffsets.push_back(totalNeurons);
        }

        weights.resize(totalWeights, 0.1f);
        biases.resize(totalBiases, 0.0f);
        neurons.resize(totalNeurons, 0.0f);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);

        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = dist(gen);
        }
        for (size_t i = 0; i < biases.size(); ++i) {
            biases[i] = 0.0f;
        }
    }

    float sigmoid(float x) {
        return 1 / (1 + exp(-x));
    }

    float relu(float x) {
        return max(0.0f, x);
    }

    // Tangente hyperbolique
    float tanhActivation(float x) {
        return tanh(x);
    }

    // Fonction d'activation linéaire
    float linear(float x) {
        return x;
    }

    // Leaky ReLU
    float leakyRelu(float x) {
        return max(0.01f * x, x);
    }

    // Modification de la fonction d'activation
    float activationFunction(float x, int functionType) {
        switch (functionType) {
            case 0: return sigmoid(x);
            case 1: return relu(x);
            case 2: return tanhActivation(x);
            case 3: return linear(x);
            case 4: return leakyRelu(x);
            // Add more cases as needed
            default: return sigmoid(x);
        }
    }

    // Dérivée de la Sigmoïde (reçoit le neurone déjà calculé)
    float sigmoidDerivative(float output) {
        return output * (1.0f - output);
    }

    // Dérivée de ReLU
    float reluDerivative(float output) {
        return output > 0.0f ? 1.0f : 0.0f;
    }

    // Dérivée de Tanh (1 - tanh²) -> si 'output' est le résultat du tanh
    float tanhDerivative(float output) {
        return 1.0f - (output * output);
    }

    // Dérivée linéaire
    float linearDerivative(float output) {
        return 1.0f;
    }

    // Dérivée Leaky ReLU
    float leakyReluDerivative(float output) {
        return output > 0.0f ? 1.0f : 0.01f;
    }

    // Dispatcher des dérivées
    float activationFunctionDerivative(float output, int functionType) {
        switch (functionType) {
            case 0: return sigmoidDerivative(output);
            case 1: return reluDerivative(output);
            case 2: return tanhDerivative(output);
            case 3: return linearDerivative(output);
            case 4: return leakyReluDerivative(output);
            default: return sigmoidDerivative(output);
        }
    }

    std::vector<float> feedForward(const std::vector<float>& input) {
        // Copie directe des entrées dans la mémoire continue des neurones
        std::copy(input.begin(), input.end(), neurons.begin() + neuronOffsets[0]);

        for (size_t l = 0; l < layerSizes.size() - 1; ++l) {
            int inSize = layerSizes[l];
            int outSize = layerSizes[l + 1];
            
            size_t wStart = weightOffsets[l];
            size_t bStart = biasOffsets[l];
            size_t inStart = neuronOffsets[l];
            size_t outStart = neuronOffsets[l + 1];

            // Multiplication matricielle optimisée en cache CPU
            for (int j = 0; j < outSize; ++j) {
                float sum = biases[bStart + j];
                for (int i = 0; i < inSize; ++i) {
                    sum += neurons[inStart + i] * weights[wStart + i * outSize + j];
                }
                neurons[outStart + j] = activationFunction(sum, layerActivationF[l]);
            }
        }

        // Récupération de la dernière couche (les outputs)
        size_t lastLayerIdx = layerSizes.size() - 1;
        size_t outStart = neuronOffsets[lastLayerIdx];
        int outSize = layerSizes[lastLayerIdx];

        return std::vector<float>(neurons.begin() + outStart, neurons.begin() + outStart + outSize);
    }

    std::vector<float> backPropagation(const std::vector<float>& target, float learningRate) {
        size_t numLayers = layerSizes.size();
        std::vector<float> deltas(neurons.size(), 0.0f);

        // 1. Couche de sortie
        size_t lastLayerIdx = numLayers - 1;
        size_t outStart = neuronOffsets[lastLayerIdx];
        int outSize = layerSizes[lastLayerIdx];
        int lastActType = layerActivationF[numLayers - 2];

        for (int j = 0; j < outSize; ++j) {
            float prediction = neurons[outStart + j];
            float error = prediction - target[j];
            float derivative = activationFunctionDerivative(prediction, lastActType);
            deltas[outStart + j] = error * derivative;
        }

        // 2. Couches cachées (de l'avant-dernière couche jusqu'à 0)
        for (int l = numLayers - 2; l >= 0; --l) {
            int inSize = layerSizes[l];
            int outSize = layerSizes[l + 1];
            
            size_t wStart = weightOffsets[l];
            size_t bStart = biasOffsets[l];
            size_t inStart = neuronOffsets[l];
            size_t nextStart = neuronOffsets[l + 1];
            int actType = layerActivationF[l]; // La bonne fonction d'activation pour cette couche

            // Mise à jour des poids et des biais pour cette couche
            for (int j = 0; j < outSize; ++j) {
                float d = deltas[nextStart + j];
                biases[bStart + j] -= learningRate * d;

                for (int i = 0; i < inSize; ++i) {
                    float prevActivation = neurons[inStart + i];
                    weights[wStart + i * outSize + j] -= learningRate * d * prevActivation;
                }
            }

            // Propagation de l'erreur vers la couche précédente (uniquement si ce n'est pas l'entrée pure)
            if (l > 0) {
                for (int i = 0; i < inSize; ++i) {
                    float errorSum = 0.0f;
                    for (int j = 0; j < outSize; ++j) {
                        errorSum += deltas[nextStart + j] * weights[wStart + i * outSize + j];
                    }
                    float derivative = activationFunctionDerivative(neurons[inStart + i], layerActivationF[l - 1]);
                    deltas[inStart + i] = errorSum * derivative;
                }
            }
        }

        int inSize = layerSizes[0];
        outSize = layerSizes[1];
        size_t wStart = weightOffsets[0];
        size_t nextStart = neuronOffsets[1];

        std::vector<float> inputGradients(inSize, 0.0f);
        for (int i = 0; i < inSize; ++i) {
            float errorSum = 0.0f;
            for (int j = 0; j < outSize; ++j) {
                errorSum += deltas[nextStart + j] * weights[wStart + i * outSize + j];
            }
            inputGradients[i] = errorSum; // C'est ce qu'on renvoie à l'attention !
        }

        return inputGradients;
    }

    void saveNetwork(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary); // Optionnel : texte ou binaire, ici on garde du texte simple
        if (!file.is_open()) return;

        // 1. Sauvegarde de la structure (nombre de couches, puis les tailles)
        file << layerSizes.size() << "\n";
        for (int size : layerSizes) {
            file << size << " ";
        }
        file << "\n";

        // 2. Sauvegarde des fonctions d'activation par couche
        for (int act : layerActivationF) {
            file << act << " ";
        }
        file << "\n";

        // 3. Sauvegarde de tous les poids d'un coup
        file << weights.size() << "\n";
        for (float w : weights) {
            file << w << " ";
        }
        file << "\n";

        // 4. Sauvegarde de tous les biais d'un coup
        file << biases.size() << "\n";
        for (float b : biases) {
            file << b << " ";
        }
        file << "\n";
    }

    void loadNetwork(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) return;

        // 1. Chargement de la structure
        size_t numLayers;
        file >> numLayers;
        layerSizes.resize(numLayers);
        for (size_t i = 0; i < numLayers; ++i) {
            file >> layerSizes[i];
        }

        // 2. Chargement des fonctions d'activation
        layerActivationF.resize(numLayers > 0 ? numLayers - 1 : 0);
        for (size_t i = 0; i < layerActivationF.size(); ++i) {
            file >> layerActivationF[i];
        }

        // 3. Chargement des poids
        size_t weightsSize;
        file >> weightsSize;
        weights.resize(weightsSize);
        for (size_t i = 0; i < weightsSize; ++i) {
            file >> weights[i];
        }

        // 4. Chargement des biais
        size_t biasesSize;
        file >> biasesSize;
        biases.resize(biasesSize);
        for (size_t i = 0; i < biasesSize; ++i) {
            file >> biases[i];
        }
    }
};