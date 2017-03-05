package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"github.com/montanaflynn/stats"
)

type Classifier interface {
	Fit(X mat64.Matrix, Y mat64.Vector)
	Predict(X mat64.Matrix)
	Score(X mat64.Matrix)
}

func main() {
	fmt.Println("Hello World")

	nums := []float64{1, 2, 1}
	fmt.Println(stats.Mean(nums))

	x := indicesOf(1, nums)
	fmt.Println(x)
}
