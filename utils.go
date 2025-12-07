package main

import (
	"encoding/csv"
	"fmt"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/lintrans"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
	"math"
	"math/big"
	"os"
	"strconv"
)

func readX(filename string) ([][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	data := make([][]float64, len(records))
	for i, row := range records {
		data[i] = make([]float64, len(row))
		for j, val := range row {
			num, err := strconv.ParseFloat(val, 64)
			if err != nil {
				return nil, fmt.Errorf("The parsing failed at row %d, column %d: %v", i+1, j+1, val)
			}
			data[i][j] = num
		}
	}

	return data, nil
}

func ready(filename string) ([]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var result []float64
	for i, row := range records {
		for j, val := range row {
			num, err := strconv.ParseFloat(val, 64)
			if err != nil {
				return nil, fmt.Errorf("The parsing failed at row %d, column %d: %v", i+1, j+1, val)
			}
			result = append(result, num)
		}
	}

	return result, nil
}

func sigmod(params ckks.Parameters, K float64, eval *ckks.Evaluator,
	ct *rlwe.Ciphertext) (ct_sigmod *rlwe.Ciphertext) {

	var err error
	sigmoid := func(x float64) (y float64) {
		return 1 / (math.Exp(-x) + 1)
	}

	// Chebyhsev approximation of the sigmoid in the domain [-K, K] of degree 63.
	poly := polynomial.NewPolynomial(GetChebyshevPoly(K, 63, sigmoid))

	// Instantiates the polynomial evaluator
	polyEval := polynomial.NewEvaluator(params, eval)

	// Retrieves the change of basis y = scalar * x + constant
	scalar, constant := poly.ChangeOfBasis()

	// Performes the change of basis Standard -> Chebyshev
	if err := eval.MulRelin(ct, scalar, ct); err != nil {
		panic(err)
	}

	if err := eval.Add(ct, constant, ct); err != nil {
		panic(err)
	}

	if err := eval.Rescale(ct, ct); err != nil {
		panic(err)
	}

	// Evaluates the polynomial
	if ct_sigmod, err = polyEval.Evaluate(ct, poly, params.DefaultScale()); err != nil {
		panic(err)
	}

	return ct_sigmod
}

// GetChebyshevPoly returns the Chebyshev polynomial approximation of f the
// in the interval [-K, K] for the given degree.
func GetChebyshevPoly(K float64, degree int, f64 func(x float64) (y float64)) bignum.Polynomial {

	FBig := func(x *big.Float) (y *big.Float) {
		xF64, _ := x.Float64()
		return new(big.Float).SetPrec(x.Prec()).SetFloat64(f64(xF64))
	}

	var prec uint = 128

	interval := bignum.Interval{
		A:     *bignum.NewFloat(-K, prec),
		B:     *bignum.NewFloat(K, prec),
		Nodes: degree,
	}

	// Returns the polynomial.
	return bignum.ChebyshevApproximation(FBig, interval)
}

func HomomoMatMutiVec(rlk *rlwe.RelinearizationKey, kgen *rlwe.KeyGenerator, sk *rlwe.SecretKey,
	ctX *rlwe.Ciphertext, cty *rlwe.Ciphertext, params ckks.Parameters, Slots int,
	eval *ckks.Evaluator, ecd *ckks.Encoder, enc *rlwe.Encryptor,
	k int, evalInnsum *ckks.Evaluator, batch int) (ct_MVM *rlwe.Ciphertext) {

	var err error

	rot := -k
	galElsRot := []uint64{
		params.GaloisElement(rot)}
	rotEval := eval.WithKey(rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(galElsRot, sk)...))

	vec0 := make([]float64, Slots)
	ptVec0 := ckks.NewPlaintext(params, params.MaxLevel())
	if err = ecd.Encode(vec0, ptVec0); err != nil {
		panic(err)
	}
	ctVec0, err := enc.EncryptNew(ptVec0)
	if err != nil {
		panic(err)
	}

	vec00 := make([]float64, Slots)
	ptVec00 := ckks.NewPlaintext(params, params.MaxLevel())
	if err = ecd.Encode(vec00, ptVec00); err != nil {
		panic(err)
	}
	ctVec00, err := enc.EncryptNew(ptVec00)
	if err != nil {
		panic(err)
	}

	vec1 := make([]float64, k) 
	if k > 0 {
		vec1[0] = 1.0
	}
	ptVec1 := ckks.NewPlaintext(params, params.MaxLevel())
	if err = ecd.Encode(vec1, ptVec1); err != nil {
		panic(err)
	}
	ctVec1, err := enc.EncryptNew(ptVec1)
	if err != nil {
		panic(err)
	}
	for i := 0; i < k; i++ {
		ctVec0, err = eval.AddNew(ctVec0, cty)
		if err != nil {
			panic(err)
		}
		cty, err = rotEval.RotateNew(cty, rot)
	}

	ctX, err = eval.MulRelinNew(ctX, ctVec0)
	if err != nil {
		panic(err)
	}

	if err = eval.Rescale(ctX, ctX); err != nil {
		panic(err)
	}

	if err := evalInnsum.InnerSum(ctX, batch, k, ctX); err != nil {
		panic(err)
	}

	ctMVM := make([]*rlwe.Ciphertext, k)

	for i := 0; i < k; i++ {
		ctMVM[i], err = eval.MulRelinNew(ctX, ctVec1)
		if err != nil {
			panic(err)
		}
		if err = eval.Rescale(ctMVM[i], ctMVM[i]); err != nil {
			panic(err)
		}
		ctVec1, err = rotEval.RotateNew(ctVec1, rot)
	}

	ctVec00, err = eval.AddNew(ctVec00, ctMVM[0])
	if err != nil {
		panic(err)
	}

	for i := 1; i < k; i++ {
		rot1 := (k - i) + k*(i-1)
		galElsRot1 := []uint64{
			params.GaloisElement(rot1)}
		rotEval1 := eval.WithKey(rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(galElsRot1, sk)...))

		ctMVM[i], err = rotEval1.RotateNew(ctMVM[i], rot1)

		ctVec00, err = eval.AddNew(ctVec00, ctMVM[i])
		if err != nil {
			panic(err)
		}
	}
	ct_MVM = ctVec00

	return ct_MVM
}

func HomomoMatMutiVec1(lt lintrans.LinearTransformation, ltEval *lintrans.Evaluator,
	ctVec *rlwe.Ciphertext, eval *ckks.Evaluator, n int, LogN int, ctVec0 *rlwe.Ciphertext,
	rotEval *ckks.Evaluator, rot int) (ctLintransVec *rlwe.Ciphertext) {

	var err error

	logNPow := math.Pow(2, float64(LogN-1))
	
	logNPown := logNPow/float64(n) - 1

	for i := 0; i < int(logNPown); i++ {
		ctVec0, err = eval.AddNew(ctVec, ctVec0)
		if err != nil {
			panic(err)
		}
		ctVec, err = rotEval.RotateNew(ctVec, rot)
		if err != nil {
			panic(err)
		}
	}
	ctVec = ctVec0

	if err := ltEval.Evaluate(ctVec, lt, ctVec); err != nil {
		panic(err)
	}

	if err = eval.Rescale(ctVec, ctVec); err != nil {
		panic(err)
	}

	ctLintransVec = ctVec

	return ctLintransVec
}
