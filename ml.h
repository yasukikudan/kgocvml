#ifdef __cplusplus
extern "C" {
#endif

typedef void* GOMat;
GOMat NewGOMat(int r,int c);
float GOMatGet(GOMat t,int r,int c);
void  GOMatSet(GOMat t,int r,int c,float v);
int   GOMatRow(GOMat t);
int   GOMatColunm(GOMat t);

typedef void* GONeuralNetwork;
GONeuralNetwork NewNeuralNetwork(int *layer,int layercount);
void GONeuralNetworkTrain(GONeuralNetwork,GOMat,GOMat);
GOMat GONeuralNetworkPredict(GONeuralNetwork,GOMat);
int add(int x,int y);

#ifdef __cplusplus
}
#endif
