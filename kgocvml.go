// kgocvml project kgocvml.go
package kgocvml

/*
#cgo CFLAGS:-mavx -std=c11 -Wno-pointer-to-int-cast
#cgo CXXFLAGS:-mavx -std=c++11
#cgo LDFLAGS: -lrt -lstdc++  -L/usr/lib/x86_64-linux-gnu -lopencv_objdetect    -lopencv_calib3d  -lopencv_imgproc -lopencv_stitching -lopencv_core -lopencv_superres -lopencv_features2d -lopencv_ml -lopencv_ts -lopencv_flann  -lopencv_video -lopencv_highgui -lopencv_photo -lopencv_videostab
#include "ml.h"
*/
import "C"

import (
	"fmt"
)

// cv::mat_<float>型
type Mat struct {
	mat C.GOMat
	//Row()    int
	//Colunm() int
}

func ToMat(b [][]float64) Mat {
	if len(b) <= 0 {
		panic(" slice empty")
	} else {
		if len(b[0]) <= 0 {
			panic(" slice empty")
		}
	}
	r := len(b)
	c := len(b[0])
	out := NewMat(r, c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out.Set(i, j, b[i][j])
		}
	}
	return out
}

func NewMat(r, c int) Mat {
	var m Mat
	m.mat = C.NewGOMat(C.int(r), C.int(c))
	//m.Row() = r
	//m.Colunm() = c
	return m
}

func (m *Mat) Row() int {
	//return m.Row()
	return int(C.GOMatRow(m.mat))
}

func (m *Mat) Colunm() int {
	//return m.Colunm()

	return int(C.GOMatColunm(m.mat))
}

func (m *Mat) Set(r, c int, v float64) {
	if r < 0 || m.Row() <= r {
		panic("index out of range ")
	}
	if c < 0 || m.Colunm() <= c {
		panic("index out of range ")
	}
	C.GOMatSet(m.mat, C.int(r), C.int(c), C.float(v))
}

func (m *Mat) Get(r, c int) float64 {
	if r < 0 || m.Row() <= r {
		panic("index out of range ")
	}
	if c < 0 || m.Colunm() <= c {
		panic("index out of range ")
	}
	return float64(C.GOMatGet(m.mat, C.int(r), C.int(c)))
}

func (m *Mat) FromMat() [][]float64 {
	fmt.Println("frommat ", m.Row(), m.Colunm())
	out := make([][]float64, m.Row())
	for i := 0; i < m.Row(); i++ {
		tmp := []float64{}
		for j := 0; j < m.Colunm(); j++ {
			tmp = append(tmp, m.Get(i, j))
		}
		out[i] = tmp
	}
	return out
}

//
type NeuralNetwork struct {
	nn C.GONeuralNetwork
}

func NewNeuralNetwork(layers []int32) NeuralNetwork {
	var out NeuralNetwork
	out.nn = C.NewNeuralNetwork((*C.int)(&layers[0]), C.int(len(layers)))
	return out
}

func (n *NeuralNetwork) Train(t, r Mat) {
	C.GONeuralNetworkTrain(n.nn, t.mat, r.mat)
}
func (n *NeuralNetwork) Predict(t Mat) Mat {
	var out Mat
	out.mat = C.GONeuralNetworkPredict(n.nn, t.mat)
	fmt.Println(out.Row(), out.Colunm(), out.Get(0, 0))
	return out
}

func Ok() {
	fmt.Println("ahflkh")
	mat := NewMat(1, 100)
	mat.Set(0, 50, 1000.0002)
	fmt.Println(mat.Get(0, 50))
}
