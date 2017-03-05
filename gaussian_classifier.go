package main

import "github.com/gonum/matrix/mat64"

type LDA struct {
	Means  []float64
	Cov    *mat64.Dense
	Priors []float64
}

func indicesOf(val float64, data []float64) []int {
	indices := make([]int, 0)
	for idx, v := range data {
		if v == val {
			indices = append(indices, idx)
		}
	}
	return indices
}

func trainDataAtIdices(indices []int, trainData *mat64.Dense) *mat64.Dense {
	_, c := trainData.Dims()
	temp := make([]float64, len(indices)*c, len(indices)*c)
	for idx := range indices {
		begin, end := idx*c, (idx+1)*c
		mat64.Row(temp[begin:end], idx, trainData)
	}
	return mat64.NewDense(len(indices), c, temp)
}

func getGaussianParams(trainData *mat64.Dense, trainLabs []float64, allLabs []float64) (*mat64.Dense, []float64, []float64) {
	numClasses := len(allLabs)
	covMatrices := make([]mat64.Dense, numClasses, numClasses)
	means := make([]float64, numClasses, numClasses)
	for i := 0; i < numClasses; i++ {
		idx := indicesOf(allLabs[i], trainLabs)
		data := trainDataAtIdices(idx, *trainData)
		// TODO: calculate cov matrix + mus + priors
	}
	return nil, nil, nil
}
