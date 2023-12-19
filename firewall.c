#include <stdio.h>
#include <stdlib.h>
#include <svm.h>

int main(int argc, char* argv[]) {
    // Load the trained model and vectorizer
    struct svm_model* model = svm_load_model("firewall_model.joblib.model");
    struct svm_node* x;
    struct svm_problem test_prob;
    struct svm_node* test_x[1];
    struct svm_parameter param;
    int max_nr_attr = 1024;

    // Initialize the SVM parameters
    param.svm_type = C_SVC;
    param.kernel_type = LINEAR;
    param.degree = 3;
    param.gamma = 0.5;
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 1;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;

    // Load the URL to be classified
    const char* url = "http://example.com";
    
    // Transform the input URL into numerical features
    // You need to implement this part based on your vectorizer
    // Create a test_x array with the numerical features
    
    // Inference
    double predict_label = svm_predict(model, test_x[0]);

    // Output the result
    if (predict_label == 1) {
        printf("Malicious URL.\n");
    } else {
        printf("Not malicious URL.\n");
    }

    // Clean up
    free(x);
    svm_free_and_destroy_model(&model);
    svm_destroy_param(&param);

    return 0;
}
