#include <vector>
#include <random>
#include <iostream>
#include <stdexcept>

class Embedding {
private:
    int vocabSize;
    int hiddenDim;
    std::vector<float> weights;

public:
    Embedding(int vocabSize, int hiddenDim) 
        : vocabSize(vocabSize), hiddenDim(hiddenDim), weights(vocabSize * hiddenDim) {
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.02f);
        
        for (auto& w : weights) {
            w = dist(gen);
        }
    }

    std::vector<float> forward(int id) const {
        if (id < 0 || id >= vocabSize) {
            throw std::out_of_range("ID hors limites dans la couche d'Embedding.");
        }

        std::vector<float> vec(hiddenDim);
        int startIdx = id * hiddenDim;
        for (int i = 0; i < hiddenDim; ++i) {
            vec[i] = weights[startIdx + i];
        }
        return vec;
    }

    std::vector<std::vector<float>> forwardSequence(const std::vector<int>& ids) const {
        std::vector<std::vector<float>> output;
        output.reserve(ids.size());
        for (int id : ids) {
            output.push_back(forward(id));
        }
        return output;
    }

    void backprop(const std::vector<int>& ids, const std::vector<std::vector<float>>& gradients, float learningRate) {
        // Affichage direct pour voir si les gradients arrivent et ce qu'ils valent
        // std::cout << "[Embedding::backprop] Lignes de gradients reçues : " << gradients.size() << "\n";
        // for (size_t t = 0; t < gradients.size(); ++t) {
        //     std::cout << "  grad[" << t << "] (ID " << ids[t] << ") : ";
        //     float sum = 0.0f;
        //     for (size_t i = 0; i < gradients[t].size(); ++i) {
        //         sum += std::abs(gradients[t][i]);
        //     }
        //     std::cout << "Somme absolue = " << sum << "\n";
        // }

        for (size_t t = 0; t < ids.size(); ++t) {
            int id = ids[t];
            if (id < 0 || id >= vocabSize) continue;

            int startIdx = id * hiddenDim;
            for (int i = 0; i < hiddenDim; ++i) {
                weights[startIdx + i] -= learningRate * gradients[t][i];
            }
        }
    }

    int getHiddenDim() const { return hiddenDim; }
};