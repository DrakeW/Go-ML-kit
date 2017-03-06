package main

import (
	"github.com/gonum/stat"
	"github.com/gonum/matrix/mat64"
)

type Axis int

const (
	ROW Axis = 1
	COL Axis = 0
)


// Linear Discriminant Analysis classifier
type LDA struct {
	means  []*mat64.Vector
	cov    *mat64.SymDense
	priors *mat64.Vector
}

func indicesOf(val float64, data *[]float64) []int {
	indices := make([]int, 0)
	for idx, v := range *data {
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
	if axis == ROW {
		vec = mat64.NewVector(c, make([]float64, c, c))
		for i := 0; i < r; i++ {
			vec.AddVec(vec, m.RowView(i))
		}
		vec.ScaleVec(1/(float64(r)), vec)
	} else if axis == COL {
		vec = mat64.NewVector(c, make([]float64, r, r))
		for i := 0; i < r; i++ {
			vec.AddVec(vec, m.ColView(i))
		}
		vec.ScaleVec(1/(float64(c)), vec)
	}
	return vec
}

func calcCovMat(m *mat64.Dense) *mat64.SymDense {
	_, c := m.Dims()
	covMat := mat64.NewSymDense(c, make([]float64, c*c, c*c))
	stat.CovarianceMatrix(covMat, m, nil)
	return covMat
}

// return cov matrices, mean for each class, prior for each class
func getGaussianParams(trainData *mat64.Dense, trainLabs *[]float64, allLabs []float64) ([]*mat64.SymDense, []*mat64.Vector, *mat64.Vector) {
	numClasses := len(allLabs)
	covMatrices := make([]*mat64.SymDense, numClasses, numClasses)
	means := make([]*mat64.Vector, numClasses, numClasses)
	priors := make([]float64, numClasses, numClasses)
	for i := 0; i < numClasses; i++ {
		idx := indicesOf(allLabs[i], trainLabs)
		data := trainDataAtIdices(idx, trainData)

		priors = append(priors, float64(len(idx))/float64(len(*trainLabs)))
		means = append(means, matrixMean(data, ROW))
		covMatrices = append(covMatrices, calcCovMat(data))
	}
	priors_vec := mat64.NewVector(numClasses, priors)
	return covMatrices, means, priors_vec
}

// TODO: test above code!!
// TODO: write function of LDA that confomrs to classifier interface

func Set(xs []float64) []float64 {
	found := make(map[float64]bool)
	j := 0
	for i, x := range xs {
		if !found[x] {
			found[x] = true
			xs[j] = xs[i]
			j++
		}
	}
	return xs[:j]
}

func (l *LDA) Fit(X *mat64.Dense, Y []float64) {
	allLabs := Set(Y)
	cov_mats, means, priors := getGaussianParams(mat64.DenseCopyOf(X), &Y, allLabs)

	_, c := X.Dims()
	covM := mat64.NewSymDense(c, make([]float64, c*c, c*c))
	for c := range cov_mats {
		covM.AddSym(covM, c)
	}
	covM.ScaleSym(float64(1)/float64(len(allLabs)), covM)

	l.cov = covM
	l.means = means
	l.priors = priors
}

func (l *LDA) Predict(X *mat64.Matrix) *mat64.Vector {
	return nil
}

func (l *LDA) Score(X *mat64.Matrix) float64 {
	return 0
}
