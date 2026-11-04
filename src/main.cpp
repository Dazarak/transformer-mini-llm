#include "tokenizer.h"
#include "embedding.h"
#include "attention.h"
#include "neuralnetwork.h"

#include <iostream>
#include <vector>
#include <fstream>

// Fonction utilitaire pour créer le vecteur cible (One-Hot)
std::vector<float> createTargetVector(int targetId, int vocabSize) {
    std::vector<float> target(vocabSize, 0.0f);
    target[targetId] = 1.0f; // On met 1 sur le mot qu'il fallait deviner
    return target;
}

void addPositionalEncoding(std::vector<std::vector<float>>& embeddedSeq, int hiddenDim) {
    size_t seqLen = embeddedSeq.size();
    for (size_t pos = 0; pos < seqLen; ++pos) {
        for (int d = 0; d < hiddenDim; ++d) {
            float freq = std::pow(10000.0f, -((float)(d / 2) * 2.0f) / hiddenDim);
            float pe = (d % 2 == 0) ? std::sin(pos * freq) : std::cos(pos * freq);
            embeddedSeq[pos][d] += pe * 0.05f; // Facteur d'atténuation
        }
    }
}

int main() {
    Tokenizer tokenizer;

    // Définition d'un tout petit vocabulaire de test
    int idLe    = tokenizer.getOrAdd("Le");
    int idChat  = tokenizer.getOrAdd("chat");
    int idMange = tokenizer.getOrAdd("mange");
    int idDes   = tokenizer.getOrAdd("des");
    int idCroq  = tokenizer.getOrAdd("croquettes");

    int idLa    = tokenizer.getOrAdd("La");
    int idCrepe  = tokenizer.getOrAdd("crepe");
    int idEst = tokenizer.getOrAdd("est");
    int idAu = tokenizer.getOrAdd("au");
    int idSucre  = tokenizer.getOrAdd("sucre");

    int vocabSize = tokenizer.size();
    int hiddenDim = 16;
    float learningRate = 0.002f;
    int epochs = 20000;

    std::ofstream logFile("training_log.txt");
    if (!logFile.is_open()) {
        std::cerr << "Erreur lors de l'ouverture du fichier de log." << std::endl;
        return 1;
    }

    // Initialisation des composants
    Embedding embedding(vocabSize, hiddenDim);
    SingleHeadAttention attention(hiddenDim);
    NeuralNetwork mlp({ hiddenDim, 16, 16, vocabSize }, { 1, 1, 3 });

    // Petit dataset d'exemples (Entrée -> Cible attendue)
    struct Sample {
        std::vector<int> inputIds;
        int targetId; // L'ID du mot à prédire
    };

    std::vector<Sample> dataset = {
        {{idLe }, {idChat}},
        {{idLe, idChat }, {idMange}},
        {{idLe, idChat, idMange }, {idDes}},
        {{idLe, idChat, idMange, idDes}, {idCroq}},
        {{idLe, idChat, idMange, idDes, idCroq}, 0},
        {{idLa }, {idCrepe}},
        {{idLa, idCrepe },  {idEst}},
        {{idLa, idCrepe, idEst },  {idAu}},
        {{idLa, idCrepe, idEst, idAu},  {idSucre}},
        {{idLa, idCrepe, idEst, idAu, idSucre}, 0},
        {{idLe }, {idChat}},
        {{idLe, idChat }, {idMange}},
        {{idLe, idChat, idMange }, {idDes}},
        {{idLe, idChat, idMange, idDes}, {idCroq}},
        {{idLe, idChat, idMange, idDes, idCroq}, 0},
        {{idLa }, {idCrepe}},
        {{idLa, idCrepe },  {idEst}},
        {{idLa, idCrepe, idEst },  {idAu}},
        {{idLa, idCrepe, idEst, idAu},  {idSucre}},
        {{idLa, idCrepe, idEst, idAu, idSucre}, 0},
    };

    logFile << "Epoch,Prediction,Loss\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalEpochLoss = 0.0f;
        int totalSteps = 0;

        for (const auto& sample : dataset) {
            // Pour chaque sous-séquence (de 1 token jusqu'à la taille totale - 1)
            for (size_t t = 0; t < sample.inputIds.size() - 1; ++t) {
                // Sous-séquence d'entrée jusqu'au temps t
                std::vector<int> subInput(sample.inputIds.begin(), sample.inputIds.begin() + t + 1);
                int targetId = sample.inputIds[t + 1]; // Le mot suivant à prédire

                // 1. Forward Embedding & Attention
                auto embeddedSeq = embedding.forwardSequence(subInput);
                addPositionalEncoding(embeddedSeq, hiddenDim); 
                auto attentionOut = attention.forward(embeddedSeq);

                // 2. Le MLP prédit le mot suivant basé sur le dernier token visible à l'instant t
                auto mlpOutput = mlp.feedForward(attentionOut.back());

                // Calcul de la loss
                auto targetVector = createTargetVector(targetId, vocabSize);
                float loss = 0.0f;
                for (size_t j = 0; j < mlpOutput.size(); ++j) {
                    float err = mlpOutput[j] - targetVector[j];
                    loss += err * err;
                }
                loss /= vocabSize;
                totalEpochLoss += loss;
                totalSteps++;

                // 3. Backpropagation sur toute la séquence active
                auto mlpGrads = mlp.backPropagation(targetVector, learningRate);

                std::vector<std::vector<float>> attentionGrads(subInput.size(), std::vector<float>(hiddenDim, 0.0f));
                attentionGrads.back() = mlpGrads;

                auto embeddingGrads = attention.backprop(attentionGrads, learningRate);
                embedding.backprop(subInput, embeddingGrads, learningRate);
            }
        }

        float avgLoss = totalEpochLoss / totalSteps;
        logFile << epoch << "," << avgLoss << "\n";

        if (epoch % 50 == 0) {
            std::cout << "Epoch " << epoch << " | Loss moyenne : " << avgLoss << "\n";
        }
    }

    std::cout << "Cycle d'entraînement terminé avec succès.\n";

    // --- TEST DU MODÈLE SUR TOUTES LES SOUS-SÉQUENCES ---
    std::cout << "=== TEST DE PRÉDICTION SUR TOUT LE DATASET ===" << std::endl;

    // On extrait les séquences uniques du dataset d'origine pour tester chaque prédiction intermédiaire
    std::vector<std::vector<int>> testSequences = {
        {idLe},
        {idLe, idChat},
        {idLe, idChat, idMange},
        {idLe, idChat, idMange, idDes},
        {idLa},
        {idLa, idCrepe},
        {idLa, idCrepe, idEst},
        {idLa, idCrepe, idEst, idAu}
    };

    for (const auto& seq : testSequences) {
        auto embeddedSeq = embedding.forwardSequence(seq);
        auto attentionOut = attention.forward(embeddedSeq);
        auto mlpOutput = mlp.feedForward(attentionOut.back());

        int predictedId = std::distance(mlpOutput.begin(), std::max_element(mlpOutput.begin(), mlpOutput.end()));

        std::cout << "Entrée : ";
        for (int id : seq) {
            std::cout << tokenizer.getString(id) << " ";
        }
        std::cout << "-> Mot prédit : " << tokenizer.getString(predictedId) << std::endl;
    }

    return 0;
}