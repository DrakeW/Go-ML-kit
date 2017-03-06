package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"github.com/montanaflynn/stats"
)

type Classifier interface {
	Fit(X *mat64.Matrix, Y []float64)
	Predict(X *mat64.Matrix) *mat64.Vector
	Score(X *mat64.Matrix) float64
}

func main() {
	fmt.Println("Hello World")

	nums := []float64{1, 2, 1}
	fmt.Println(stats.Mean(nums))
}
