package main

import "github.com/gonum/matrix/mat64"

type Axis int

const (
	ROW Axis = 0
	COL Axis = 1
)

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

// aixs = 0 -> row mean, axis = 1 -> col mean
func matrixMean(m *mat64.Dense, axis Axis) *mat64.Vector {
	r, c := m.Dims()
	var vec *mat64.Vector
	temp := make([]float64, c*r, c*r)
	if axis == ROW {
		vec = mat64.NewVector(c, make([]float64, c, c))
		for i := 0; i < r; i++ {
			mat64.Row(temp[i*c:(i+1)*c], i, m)
			vec.AddVec(vec, mat64.NewVector(c, temp[i*c:(i+1)*c]))
		}
		vec.ScaleVec(1/(float64(r)), vec)
	} else if axis == COL {
		vec = mat64.NewVector(c, make([]float64, r, r))
		for i := 0; i < r; i++ {
			mat64.Col(temp[i*r:(i+1)*r], i, m)
			vec.AddVec(vec, mat64.NewVector(c, temp[i*r:(i+1)*r]))
		}
		vec.ScaleVec(1/(float64(c)), vec)
	}
	return vec
}

// return cov matrix, mean for each class, prior for each class
func getGaussianParams(trainData *mat64.Dense, trainLabs []float64, allLabs []float64) (*mat64.Dense, []*mat64.Vector, *mat64.Vector) {
	numClasses := len(allLabs)
	covMatrices := make([]*mat64.Dense, numClasses, numClasses)
	means := make([]*mat64.Vector, numClasses, numClasses)
	priros := make([]float64, numClasses, numClasses)
	for i := 0; i < numClasses; i++ {
		idx := indicesOf(allLabs[i], trainLabs)
		data := trainDataAtIdices(idx, trainData)

		priros = append(priros, float64(len(idx))/float64(len(trainLabs)))
		means = append(means, matrixMean(data, ROW))
		// TODO: calculate cov matrix
	}
	return nil, nil, nil
}
