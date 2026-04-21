// NOTE: Requires Dlib library to compile.
// Install Dlib and link using -ldlib flag.

#include <dlib/data_io.h>
#include <dlib/matrix.h>
#include <dlib/svm.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <random>

using namespace dlib;

// Load features from "features.csv"
std::vector<matrix<double>> load_features(const std::string& filename) {
    std::vector<matrix<double>> samples;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open features file: " + filename);
    }
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        matrix<double> sample(2, 1);
        size_t idx = 0;
        while (std::getline(ss, value, ',') && idx < 2) {
            sample(idx++) = std::stod(value);
        }
        samples.push_back(sample);
    }
    return samples;
}

// Load labels from "labels_row.csv"
std::vector<size_t> load_labels(const std::string& filename) {
    std::vector<size_t> labels;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open labels file: " + filename);
    }
    std::string line;
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            labels.push_back(static_cast<size_t>(std::stod(value)));
        }
    }
    return labels;
}

int main() {
    try {
        std::vector<matrix<double>> features = load_features("features.csv");
        std::vector<size_t> labels = load_labels("labels_row.csv");

        if (features.size() != labels.size()) {
            std::cerr << "Error: Number of samples (" << features.size()
                      << ") does not match number of labels (" << labels.size() << ")!" << std::endl;
            return -1;
        }

        std::default_random_engine rng(std::random_device{}());
        std::vector<std::pair<matrix<double>, size_t>> data;
        for (size_t i = 0; i < features.size(); ++i) {
            data.emplace_back(features[i], labels[i]);
        }
        std::shuffle(data.begin(), data.end(), rng);

        double train_ratio = 0.8;
        size_t num_train = static_cast<size_t>(train_ratio * data.size());
        std::vector<matrix<double>> train_features(num_train);
        std::vector<double> train_labels(num_train);
        std::vector<matrix<double>> test_features(data.size() - num_train);
        std::vector<size_t> test_labels(data.size() - num_train);

        for (size_t i = 0; i < num_train; ++i) {
            train_features[i] = data[i].first;
            train_labels[i] = (data[i].second == 1) ? 1.0 : -1.0;
        }
        for (size_t i = num_train; i < data.size(); ++i) {
            test_features[i - num_train] = data[i].first;
            test_labels[i - num_train] = data[i].second;
        }

        typedef decision_function<linear_kernel<matrix<double>>> dec_funct;
        typedef normalized_function<dec_funct> norm_funct;

        svm_c_trainer<linear_kernel<matrix<double>>> trainer;
        trainer.set_c(10);
        dec_funct df = trainer.train(train_features, train_labels);

        vector_normalizer<matrix<double>> normalizer;
        normalizer.train(train_features);

        norm_funct classifier;
        classifier.function = df;
        classifier.normalizer = normalizer;

        std::vector<size_t> predictions(test_features.size());
        for (size_t i = 0; i < test_features.size(); ++i) {
            double pred = classifier(test_features[i]);
            predictions[i] = (pred > 0) ? 1 : 0;
        }

        size_t correct = 0;
        for (size_t i = 0; i < test_labels.size(); ++i) {
            if (predictions[i] == test_labels[i]) {
                ++correct;
            }
        }
        double accuracy = static_cast<double>(correct) / test_labels.size();

        std::cout << "\nSVM model has been trained. Following are the evaluation metrics:\n";
        std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
