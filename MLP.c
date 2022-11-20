#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define learning_rate 0.01
#define nodeNum 6
#define epochNum 100000

double equation(double x){
    // double euler;
    // unsigned int n = 65535;
    // euler = pow(1 + 1.0/n, n);
    double equation_y = exp(-x)*cos(3*x);
    return equation_y;
}

double sigmoid_function(double z){
    double euler, sigmoid;
    unsigned int n=65535;
    euler = pow(1 + 1.0/n, n);
    sigmoid = 1.0 / (1.0 + exp(-z));
    //sigmoid = 1.0 / (1.0 + pow(euler, -z))
    return sigmoid;
}
double sigmoidDiff_function(double z){
    return sigmoid_function(z)*(1-sigmoid_function(z));
}

double tanh_function(double z){
    double tanh;
    tanh = (1 - exp(-2*z)) / (1 + exp(-2*z));
    return tanh;
}
double tanhDiff_function(double z){
    double tanhD;
    tanhD = 1 - pow(tanh_function(z), 2);
    // tanhD = (4*exp(-2*tanh))/(1+exp(-2*tanh))
    return tanhD;
}

void forward_propagation(double x, double vik[], double wkj[], double bias1[], double bias2[], double s[], double h[], double *z, double *y){
    for(int k=0; k<nodeNum; k++){
        s[k] = (vik[k] * x) + bias1[k];
        h[k] = tanh_function(s[k]);
    }
    double zj = 0;
    for(int k=0; k<nodeNum; k++){
        zj += (wkj[k] * h[k]) + bias2[k];
    }
    *z = zj;
    *y = tanh_function(zj);
}

void backward_propagation_wkj(double x, double y, double label, double vik[], double wkj[], double bias1[], double bias2[], double h[], double predict_z){
    for(int k=0; k<nodeNum; k++){
        double deltaWkj, deltaBias2;
        deltaWkj = ((y-label) * tanhDiff_function(predict_z) * h[k]);
        deltaBias2 = (y-label) * tanhDiff_function(predict_z);
        wkj[k] = wkj[k] - (learning_rate * deltaWkj);
        bias2[k] = bias2[k] - (learning_rate * deltaBias2);
    }
}
void backward_propagation_vik(double x, double y, double label, double vik[], double wkj[], double bias1[], double bias2[], double s[], double predict_z){
    for(int k=0; k<nodeNum; k++){
        double deltaVik, deltaBias1;
        deltaVik = ((y-label) * tanhDiff_function(predict_z) * wkj[k] * tanhDiff_function(s[k]) * x);
        deltaBias1 = (y-label) * tanhDiff_function(predict_z) * wkj[k] * tanhDiff_function(s[k]);
        vik[k] = vik[k] - (learning_rate * deltaVik);
        bias1[k] = bias1[k] - (learning_rate * deltaBias1);
    }
}

int main(){
    double x[] = {0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0};
    double label[21];
    int i, k, j, epoch;

    for(i=0; i<21; i++){    // Label Dataset
        label[i] = equation(x[i]);
    }
    
    // 先給vik & wkj兩種權重 亂數
    double vik[nodeNum], wkj[nodeNum], bias1[nodeNum], bias2[nodeNum], rd1, rd2, rd3, rd4;
    srand((unsigned)time(NULL));
    for(i=0; i<nodeNum; i++){
        rd1 = (double)rand() / (RAND_MAX + 1.0);
        vik[i] = rd1;
        rd2 = (double)rand() / (RAND_MAX + 1.0);
        wkj[i] = rd2;
        rd3 = (double)rand() / (RAND_MAX + 1.0);
        bias1[i] = rd3;
        rd4 = (double)rand() / (RAND_MAX + 1.0);
        bias2[i] = rd4;
    }
    
    //  訓練與測試
    printf("-------MLP-------\n");
    double s[nodeNum]={0.0}, h[nodeNum]={0.0}, predict_z, predict_y;
    for(epoch=0; epoch<epochNum; epoch++){
        for(i=0; i<21; i++){
            forward_propagation(x[i], vik, wkj, bias1, bias2, s, h, &predict_z, &predict_y);
            backward_propagation_wkj(x[i], predict_y, label[i], vik, wkj, bias1, bias2, h, predict_z);
            backward_propagation_vik(x[i], predict_y, label[i], vik, wkj, bias1, bias2, s, predict_z);
        }
        if(epoch % (epochNum/10) == 0){
            printf("EPOCH : %d / %d\n", epoch, epochNum);
        }
    }
    printf("Training Completed\n\n");
    
    double input, output, zz;
    while(1){
        printf("Input: ");
        scanf("%lf", &input);
        forward_propagation(input, vik, wkj, bias1, bias2, s, h, &zz, &output);
        printf("Actual Value: %f\n", equation(input));
        printf("Predict Value: %f\n\n", output);
    }

    return 0;
}