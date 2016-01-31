#include "ml.h"
#include "iostream"
#include <opencv2/ml/ml.hpp>

GOMat NewGOMat(int r,int c){
	auto out=(void*)new cv::Mat_<float>(r,c);
	return out;
}

float GOMatGet(GOMat t,int r,int c){
	auto m=(cv::Mat_<float>*)t;
	float o=m->at<float>(r,c);
	return o;
}

void GOMatSet(GOMat t,int r,int c,float v){
	auto m=(cv::Mat_<float>*)t;
	m->at<float>(r,c)=v;
}

int GOMatColunm(GOMat t){
	auto m=(cv::Mat_<float>*)t;
	return m->cols;
}

int GOMatRow(GOMat t){
	auto m=(cv::Mat_<float>*)t;
	return m->rows;
}

cv::Mat ConvertGoSlicefloat64ToMat(double* p,int len){
	auto out=cv::Mat_<float>(1,len);
	for(auto i=0;i<len;i++){
		out.at<float>(0,i)=static_cast<double>(p[i]);
	}
}

struct GONeuralNetwork_Base{
	cv::NeuralNet_MLP neuron;
	cv::ANN_MLP_TrainParams params;
};

GONeuralNetwork NewNeuralNetwork(int *layer,int layercount){
	//レイヤー構成を読み込む
	cv::Mat layerlayout(layercount,1,CV_32SC1);
	for(auto i=0;i<layercount;i++){
		layerlayout.at<int>(i)=layer[i];
	}
	auto out=new GONeuralNetwork_Base();
    out->neuron.create(layerlayout,CvANN_MLP::SIGMOID_SYM,0.6,1.0);
	//繰り返し条件
	cv::TermCriteria tcri;
	tcri.type=cv::TermCriteria::COUNT;
	tcri.maxCount=100;
	//パラメータの設定
	out->params.term_crit=tcri;
	out->params.train_method=cv::ANN_MLP_TrainParams::RPROP;
	out->params.rp_dw0=0.1;
	out->params.rp_dw_min=FLT_EPSILON;
	return (void*)out;
	//neuron.train(traindata,teacherdata,cv::Mat(),cv::Mat(),params);
}

void GONeuralNetworkTrain(GONeuralNetwork n,GOMat traindata,GOMat teacherdata){
	auto nn=(GONeuralNetwork_Base*)n;
	auto train=(cv::Mat*)traindata;
	auto teacher=(cv::Mat*)teacherdata;
    nn->neuron.train(*train,*teacher,cv::Mat(),cv::Mat(),nn->params);
}
GOMat GONeuralNetworkPredict(GONeuralNetwork n,GOMat traindata){
	auto nn=(GONeuralNetwork_Base*)n;
	auto train=(cv::Mat*)traindata;

	cv::Mat& t=*train;
	cv::Mat result;
	nn->neuron.predict(t,result);
	cv::Mat* out=new cv::Mat();
	out->operator =(result);
	//std::cout << r.rows << r.cols << "\n";
	return out;
	/*
	for(auto i=0;i<8;i++){
		std::cout << i;
		auto d=train->row(i);
		nn->neuron.predict(d,results);
		for(auto j=0;j<d.cols;j++){
			std::cout << " " << d.at<float>(0,j);
		}
		for(auto j=0;j<results.cols;j++){
			std::cout << " " << results.at<float>(0,j);
		}
		std::cout << "\n";
	}*/
}

int add(int x,int y){
	cv::Mat layerlayout(3,1,CV_32SC1);
    layerlayout.at<int>(0)=3;
    layerlayout.at<int>(1)=10;
    layerlayout.at<int>(2)=8;

    cv::Mat traindata=(cv::Mat_<float>(8,3) <<
                       0,0,0,
                       0,0,1,
                       0,1,0,
                       0,1,1,
                       1,0,0,
                       1,0,1,
                       1,1,0,
                       1,1,1
                       );
    cv::Mat teacherdata=(cv::Mat_<float>(8,8) <<
                         10,0,0,0,0,0,0,0,
                         0,10,0,0,0,0,0,0,
                         0,0,10,0,0,0,0,0,
                         0,0,0,10,0,0,0,0,
                         0,0,0,0,10,0,0,0,
                         0,0,0,0,0,10,0,0,
                         0,0,0,0,0,0,10,0,
                         0,0,0,0,0,0,0,10
                         );

    cv::NeuralNet_MLP neuron;
    neuron.create(layerlayout,CvANN_MLP::SIGMOID_SYM,0.6,1.0);
    //繰り返し条件
    cv::TermCriteria tcri;
    tcri.type=cv::TermCriteria::COUNT;
    tcri.maxCount=100;
    //パラメータの設定
    cv::ANN_MLP_TrainParams params;
    params.term_crit=tcri;
    params.train_method=cv::ANN_MLP_TrainParams::RPROP;
    params.rp_dw0=0.1;
    params.rp_dw_min=FLT_EPSILON;
    neuron.train(traindata,teacherdata,cv::Mat(),cv::Mat(),params);

    cv::Mat results;
    for(auto i=0;i<8;i++){
        std::cout << i;
        auto d=traindata.row(i);
        neuron.predict(d,results);
        for(auto j=0;j<3;j++){
            std::cout << " " << d.at<float>(0,j);
        }
        std::cout << " : ";
        int top=0;
        float topf=0;
        for(auto j=0;j<8;j++){
            auto p=results.at<float>(0,j);
            std::cout << " " << p ;
            if(p>topf){
                topf=p;
                top=j;
            }
        }
        std::cout << ":"<< top << "\n";
    }

	std::cout << x << " " << y <<std::endl;
	return x+y;
}
