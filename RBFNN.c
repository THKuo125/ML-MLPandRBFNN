#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define learning_rate 0.1
#define nodeNum 21
#define epochNum 1000

double equation(double x){
    double equation_y = exp(-x)*cos(3*x);
    return equation_y;
}

double gaussian_function(double x, double c, double sigma){
    double euclidean = sqrt(pow(x-c, 2));
    double gaussian = exp(-(pow(euclidean, 2)) / (2*pow(sigma, 2)));
    return gaussian;
}

void forward_propagation(double x, double c[], double sigma, double Hin[], double Hout[], double Who[], double *predict_y){
    double yout = 0.0;
    for(int i=0; i<nodeNum; i++){
        Hin[i] = x;
        Hout[i] = gaussian_function(Hin[i], c[i], sigma);
        yout += Hout[i] * Who[i];
    }
    *predict_y = yout;
}

void backward_propagation(double x, double label, double c[], double sigma, double Hin[], double Hout[], double Who[], double predict_y){
    for(int i=0; i<nodeNum; i++){
        double deltaWho, deltaC, deltaSigma;
        deltaWho = (predict_y - label) * gaussian_function(Hin[i], c[i], sigma);
        Who[i] = Who[i] - (learning_rate * deltaWho);

        deltaC = ( (predict_y - label) * Who[i] * Hout[i] * (x-c[i]) ) / (pow(sigma, 2));
        c[i] = c[i] - (learning_rate * deltaC);

        deltaSigma = ( (predict_y - label) * Who[i] * Hout[i] * pow((x-c[i]), 2) ) / (pow(sigma, 3));
        sigma = sigma - (learning_rate * deltaSigma);
    }
}

int main(){
    double x[] = {0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0};
    double label[21];
    int i, k, j;

    for(i=0; i<21; i++){    // Label Dataset
        label[i] = equation(x[i]);
    }
    
    // random weight
    double Who[nodeNum], Wih[nodeNum], rd1, Hout[nodeNum]={0.0}, Hin[nodeNum], c[21], sigma=0.2, predict_y;
    srand((unsigned)time(NULL));
    for(i=0; i<nodeNum; i++){
        rd1 = (double)rand() / (RAND_MAX + 1.0);
        Who[i] = rd1;
    }
    for(j=0; j<nodeNum; j++){
        c[j] = x[j];
    }

    
    //  訓練與測試
    printf("-------RBFNN-------\n");
    for(int epoch=0; epoch<epochNum; epoch++){
        for(i=0; i<nodeNum; i++){
            forward_propagation(x[i], c, sigma, Hin, Hout, Who, &predict_y);
            backward_propagation(x[i], label[i], c, sigma, Hin, Hout, Who, predict_y);
        }
        if(epoch % (epochNum/10) == 0){
            printf("EPOCH : %d / %d\n", epoch, epochNum);
        }
    }
    printf("Training Completed\n\n");
    
    double input, output;
    while(1){
        printf("Input: ");
        scanf("%lf", &input);
        forward_propagation(input, c, sigma, Hin, Hout, Who, &output);
        printf("Actual Value: %f\n", equation(input));
        printf("Predict Value: %f\n\n", output);
    }

    return 0;
}


    // printf("Before\n");
    // for(i=0; i<21; i++){
        // printf("c[%d] : %f , sigma : %f , Hout[%d] : %f , Who[%d] : %f\n",i, c[i], sigma, i, Hout[i], i, Who[i]);
        // printf("sigma : %f\n", sigma);
    // }

    // printf("After\n");
    // for(i=0; i<21; i++){
        // printf("c[%d] : %f , sigma : %f , Hout[%d] : %f , Who[%d] : %f\n",i, c[i], i, sigma, i, Hout[i], i, Who[i]);
        // printf("sigma : %f\n", sigma);
    // }