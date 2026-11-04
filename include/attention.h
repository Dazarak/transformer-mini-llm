#pragma once

#include <vector>
#include <cmath>
#include <random>

class SingleHeadAttention {
private:
    int hiddenDim;

    std::vector<float> W_q;
    std::vector<float> W_k;
    std::vector<float> W_v;

    // États cachés pour la rétropropagation
    std::vector<std::vector<float>> last_inputs;
    std::vector<std::vector<float>> last_Q;
    std::vector<std::vector<float>> last_K;
    std::vector<std::vector<float>> last_V;
    std::vector<std::vector<float>> last_weights; // Sorties du softmax

    std::vector<float> matMul(const std::vector<float>& vec, const std::vector<float>& W, int rows, int cols) const {
        std::vector<float> out(cols, 0.0f);
        for (size_t j = 0; j < (size_t)cols; ++j) {
            for (size_t i = 0; i < (size_t)rows; ++i) {
                out[j] += vec[i] * W[i * cols + j];
            }
        }
        return out;
    }

    std::vector<float> softmax(const std::vector<float>& input) const {
        std::vector<float> output(input.size());
        float maxVal = input[0];
        for (float val : input) if (val > maxVal) maxVal = val;

        float sum = 0.0f;
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::exp(input[i] - maxVal);
            sum += output[i];
        }
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] /= sum;
        }
        return output;
    }

public:
    SingleHeadAttention(int hiddenDim) : hiddenDim(hiddenDim) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.02f);

        W_q.resize(hiddenDim * hiddenDim);
        W_k.resize(hiddenDim * hiddenDim);
        W_v.resize(hiddenDim * hiddenDim);

        for (auto& w : W_q) w = dist(gen);
        for (auto& w : W_k) w = dist(gen);
        for (auto& w : W_v) w = dist(gen);
    }

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& inputs) {
        last_inputs = inputs;
        size_t seqLen = inputs.size();

        last_Q.assign(seqLen, std::vector<float>(hiddenDim));
        last_K.assign(seqLen, std::vector<float>(hiddenDim));
        last_V.assign(seqLen, std::vector<float>(hiddenDim));

        for (size_t i = 0; i < seqLen; ++i) {
            last_Q[i] = matMul(inputs[i], W_q, hiddenDim, hiddenDim);
            last_K[i] = matMul(inputs[i], W_k, hiddenDim, hiddenDim);
            last_V[i] = matMul(inputs[i], W_v, hiddenDim, hiddenDim);
        }

        std::vector<std::vector<float>> output(seqLen, std::vector<float>(hiddenDim, 0.0f));
        last_weights.assign(seqLen, std::vector<float>(seqLen, 0.0f));
        float scale = std::sqrt(static_cast<float>(hiddenDim));

        for (size_t i = 0; i < seqLen; ++i) {
            std::vector<float> scores(seqLen, 0.0f);
            for (size_t j = 0; j < seqLen; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < hiddenDim; ++d) {
                    dot += last_Q[i][d] * last_K[j][d];
                }
                scores[j] = dot / scale;
            }

            last_weights[i] = softmax(scores);

            for (size_t j = 0; j < seqLen; ++j) {
                for (int d = 0; d < hiddenDim; ++d) {
                    output[i][d] += last_weights[i][j] * last_V[j][d];
                }
            }
        }

        return output;
    }

    std::vector<std::vector<float>> backprop(const std::vector<std::vector<float>>& d_output, float learningRate) {
        size_t seqLen = last_inputs.size();
        float scale = std::sqrt(static_cast<float>(hiddenDim));

        // Gradients par rapport à Q, K, V
        std::vector<std::vector<float>> d_Q(seqLen, std::vector<float>(hiddenDim, 0.0f));
        std::vector<std::vector<float>> d_K(seqLen, std::vector<float>(hiddenDim, 0.0f));
        std::vector<std::vector<float>> d_V(seqLen, std::vector<float>(hiddenDim, 0.0f));

        // Gradients des poids de l'attention
        std::vector<float> dW_q(W_q.size(), 0.0f);
        std::vector<float> dW_k(W_k.size(), 0.0f);
        std::vector<float> dW_v(W_v.size(), 0.0f);

        // 1. Rétropropagation à travers la somme pondérée et le softmax
        for (size_t i = 0; i < seqLen; ++i) {
            std::vector<float> d_weights(seqLen, 0.0f);

            for (size_t j = 0; j < seqLen; ++j) {
                float sum_dv = 0.0f;
                for (int d = 0; d < hiddenDim; ++d) {
                    sum_dv += d_output[i][d] * last_V[j][d];
                    d_V[j][d] += last_weights[i][j] * d_output[i][d];
                }
                d_weights[j] = sum_dv;
            }

            // Calcul du terme de somme pour le softmax
            float dot_dw_w = 0.0f;
            for (size_t k = 0; k < seqLen; ++k) {
                dot_dw_w += d_weights[k] * last_weights[i][k];
            }

            // Dérivée correcte du Softmax et division par scale
            std::vector<float> d_scores(seqLen, 0.0f);
            for (size_t j = 0; j < seqLen; ++j) {
                d_scores[j] = last_weights[i][j] * (d_weights[j] - dot_dw_w) / scale;
            }

            // Rétropropagation vers Q et K via les scores
            for (size_t j = 0; j < seqLen; ++j) {
                for (int d = 0; d < hiddenDim; ++d) {
                    d_Q[i][d] += d_scores[j] * last_K[j][d];
                    d_K[j][d] += d_scores[j] * last_Q[i][d];
                }
            }
        }

        // 2. Rétropropagation vers les matrices de poids W_q, W_k, W_v et calcul des gradients d'entrée
        std::vector<std::vector<float>> d_inputs(seqLen, std::vector<float>(hiddenDim, 0.0f));

        for (size_t i = 0; i < seqLen; ++i) {
            for (int r = 0; r < hiddenDim; ++r) {
                for (int c = 0; c < hiddenDim; ++c) {
                    dW_q[r * hiddenDim + c] += last_inputs[i][r] * d_Q[i][c];
                    dW_k[r * hiddenDim + c] += last_inputs[i][r] * d_K[i][c];
                    dW_v[r * hiddenDim + c] += last_inputs[i][r] * d_V[i][c];
                }
            }

            // Transposition implicite pour propager l'erreur vers l'embedding
            for (int r = 0; r < hiddenDim; ++r) {
                for (int c = 0; c < hiddenDim; ++c) {
                    d_inputs[i][r] += d_Q[i][c] * W_q[r * hiddenDim + c];
                    d_inputs[i][r] += d_K[i][c] * W_k[r * hiddenDim + c];
                    d_inputs[i][r] += d_V[i][c] * W_v[r * hiddenDim + c];
                }
            }
        }

        // 3. Mise à jour des poids de l'attention (descente de gradient)
        for (size_t i = 0; i < W_q.size(); ++i) {
            W_q[i] -= learningRate * dW_q[i];
            W_k[i] -= learningRate * dW_k[i];
            W_v[i] -= learningRate * dW_v[i];
        }

        return d_inputs; // Transmis ensuite à la couche d'Embedding
    }
};