package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/lintrans"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils"
	"log"
	"os"
	"strconv"
	"time"
)

var flagShort = flag.Bool("short", false, "run the example with a smaller and insecure ring degree.")

func main() {

	// =================================
	// 1.Instantiating the ckks.Parameters
	// =================================

	flag.Parse()

	LogN := 13

	if *flagShort {
		LogN -= 3
	}

	var err error
	var params ckks.Parameters
	if params, err = ckks.NewParametersFromLiteral(
		ckks.ParametersLiteral{
			LogN: LogN, // A ring degree of 2^{14}
			LogQ: []int{55, 45, 45, 45, 45, 45, 45, 45, 45, 45,
				45, 45, 45, 45, 45, 45}, // An initial prime of 55 bits and 7 primes of 45 bits
			LogP:            []int{61, 61, 61}, // The log2 size of the key-switching prime
			LogDefaultScale: 40,                // The default log2 of the scaling factor
			RingType:        ring.ConjugateInvariant,
			//Xs:              ring.Ternary{H: 192}, // The default log2 of the scaling factor
		}); err != nil {
		panic(err)
	}

	btpParametersLit := bootstrapping.ParametersLiteral{

		LogN: utils.Pointy(LogN + 1),

		LogP: []int{61, 61, 61, 61},

		Xs: params.Xs(),
	}

	btpParams, err := bootstrapping.NewParametersFromLiteral(params, btpParametersLit)
	if err != nil {
		panic(err)
	}

	if *flagShort {
		// Corrects the message ratio Q0/|m(X)| to take into account the smaller number of slots and keep the same precision
		btpParams.Mod1ParametersLiteral.LogMessageRatio += 16 - params.LogN()
	}

	//prec := params.EncodingPrecision()

	// ==============
	// 2.Key Generation
	// ==============

	// Key Generator
	kgen := rlwe.NewKeyGenerator(params)

	// Secret Key
	sk := kgen.GenSecretKeyNew()

	// Public Key
	pk := kgen.GenPublicKeyNew(sk)

	// Relinearlization Key
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// Evaluation Key
	evk := rlwe.NewMemEvaluationKeySet(rlk)

	// Samples values in [-K, K]
	K := 25.0

	// ====================
	// 3.Plaintext Generation
	// ====================
	Slots := params.MaxSlots()

	// Encoder
	ecd := ckks.NewEncoder(ckks.Parameters(params))

	// ======================
	// 4.Ciphertext Generation
	// ======================

	// Encryptor
	enc := rlwe.NewEncryptor(params, pk)

	// ==========
	// 5.Decryptor
	// ==========
	dec := rlwe.NewDecryptor(params, sk)

	// =================
	// 6.Evaluator Basics
	// =================
	eval := ckks.NewEvaluator(params, evk)

	btpEvk, _, err := btpParams.GenEvaluationKeys(sk)
	if err != nil {
		panic(err)
	}
	var btpEval *bootstrapping.Evaluator
	if btpEval, err = bootstrapping.NewEvaluator(btpParams, btpEvk); err != nil {
		panic(err)
	}

	//X := [][]float64{
	//	{1.0, 2.0, 3.0},
	//	{4.0, 1.0, 2.0},
	//	{3.0, 4.0, 1.0},
	//	{2.0, 3.0, 4.0},
	//}

	X_train, err := readX("PPRSLR/data/item250/user100/X_train.csv")
	if err != nil {
		fmt.Println("Error reading CSV:", err)
		return
	}
	//fmt.Println("X:", X)
	y_train, err := ready("PPRSLR/data/item250/user100/y_train.csv")
	if err != nil {
		fmt.Println("Error reading CSV:", err)
		return
	}

	// 分 batch
	batchSize := 64
	Xb_train, yb_train := SplitDataset(X_train, y_train, batchSize)

	batchNum := len(Xb_train)
	fmt.Printf("Divided into %d batches\n", batchNum)
	for i := 0; i < len(Xb_train); i++ {
		fmt.Printf("第 %d 个 batch: X.shape = %dx%d, y.len = %d\n", i+1, len(Xb_train[i]), len(Xb_train[i][0]), len(yb_train[i]))
	}

	rows_train := len(Xb_train[0])

	for i := 0; i < batchNum-1; i++ {
		Xb_train[i] = isSquare(Xb_train[i])
	}

	k := rows_train

	//ctX := make([]*rlwe.Ciphertext, len(Xb_train)-1)

	ctXt := make([]*rlwe.Ciphertext, len(Xb_train)-1)
	cty := make([]*rlwe.Ciphertext, len(Xb_train)-1)

	sumSizectXt := 0
	sumSizecty := 0
	for i := 0; i < batchNum-1; i++ {
		ctXt[i], cty[i] = encTrain(Xb_train[i], params, yb_train[i], enc)
		sizectXt, err := ctXt[i].MarshalBinary()
		if err != nil {
			panic(err)
		}
		sumSizectXt += len(sizectXt)

		sizecty, err := cty[i].MarshalBinary()
		if err != nil {
			panic(err)
		}
		sumSizecty += len(sizecty)

	}
	fmt.Printf("Ciphertext size in the initialization phase: %f MB\n", float64(sumSizectXt+sumSizecty)/1024.0/1024.0)

	X_test, err := readX("PPRSLR/data/item250/user100/X_test.csv")
	if err != nil {
		fmt.Println("Error reading CSV.:", err)
		return
	}

	rows_test := len(X_test)

	X_squ_test := isSquare(X_test)
	k_test := len(X_squ_test)

	rot := -k
	galElsRot := []uint64{
		params.GaloisElement(rot)}
	rotEval := eval.WithKey(rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(galElsRot, sk)...))

	batch := 1
	evalInnsum := eval.WithKey(rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(params.GaloisElementsForInnerSum(batch, k), sk)...))

	vec0 := make([]float64, Slots)
	ptVec0 := ckks.NewPlaintext(params, params.MaxLevel())
	if err = ecd.Encode(vec0, ptVec0); err != nil {
		panic(err)
	}
	ctVec0, err := enc.EncryptNew(ptVec0)
	if err != nil {
		panic(err)
	}

	// ===========
	// 7.Training
	// ===========
	fmt.Printf("========\n")
	fmt.Printf("TRAINING\n")
	fmt.Printf("========\n")
	fmt.Printf("\n")

	lt := make([]lintrans.LinearTransformation, len(Xb_train)-1)
	ltEval := make([]*lintrans.Evaluator, len(Xb_train)-1)

	for i := 0; i < batchNum-1; i++ {
		lt[i], ltEval[i] = LinearTrans(Xb_train[i], k, Slots, cty, params, ecd, eval, kgen, rlk, sk)
	}
	lt_test, ltEval_test := LinearTrans(X_squ_test, k_test, Slots, cty, params, ecd, eval, kgen, rlk, sk)

	start := time.Now()
	ct_theta, ct_bias := train(lt, ltEval, k, LogN, ctVec0, rotEval, rot, rlk, kgen, sk,
		ctXt, cty, evalInnsum, batch, Slots, eval, params, enc, btpEval, rows_train, K, batchNum)
	trainTime := time.Since(start)

	start1 := time.Now()
	ct_probs := predict(lt_test, ltEval_test, k_test, LogN, ctVec0, rotEval, rot,
		ct_theta, ct_bias, eval, params, K)
	testTime := time.Since(start1)

	sizect_probs, err := ct_probs.MarshalBinary()
	if err != nil {
		panic(err)
	}
	fmt.Printf("Ciphertext size in the recommendation phase: %f MB\n", float64(len(sizect_probs)/1024.0/1024.0))

	fmt.Println()
	fmt.Printf("The times of train: %v\n", trainTime)
	fmt.Printf("The times of test: %v\n", testTime)

	// Decrypts and print the stats about the precision.
	PrintPrecisionStats(ct_theta, ecd, dec)
	PrintPrecisionStats(ct_bias, ecd, dec)
	PrintPrecisionStats1(ct_probs, ecd, dec, rows_test)

}

// PrintPrecisionStats decrypts, decodes and prints the precision stats of a ciphertext.
func PrintPrecisionStats(ct *rlwe.Ciphertext, ecd *ckks.Encoder, dec *rlwe.Decryptor) {

	var err error

	// Decrypts the vector of plaintext values
	pt := dec.DecryptNew(ct)

	// Decodes the plaintext
	res := make([]float64, ct.Slots())
	if err = ecd.Decode(pt, res); err != nil {
		panic(err)
	}

	// Pretty prints some values
	fmt.Printf("res: ")
	for i := 0; i < 8; i++ {
		fmt.Printf("%10.7f ", res[i])
	}
	fmt.Printf("...\n")

}

func PrintPrecisionStats1(ct *rlwe.Ciphertext, ecd *ckks.Encoder, dec *rlwe.Decryptor, rows int) {

	var err error

	// Decrypts the vector of plaintext values
	pt := dec.DecryptNew(ct)

	// Decodes the plaintext
	res := make([]float64, ct.Slots())
	if err = ecd.Decode(pt, res); err != nil {
		panic(err)
	}

	result := make([]float64, rows)
	for i := 0; i < rows; i++ {
		if res[i] >= 0.5 {
			result[i] = 1.0
		} else {
			result[i] = 0.0
		}
	}

	// Pretty prints some values
	fmt.Printf("result: ")
	fmt.Println("rows", rows)
	for i := 0; i < rows; i++ {
		fmt.Printf("%1.f ", result[i])
	}
	fmt.Printf("...\n")

	file, err := os.Create("PPRSLR/data/item250/user100/output.csv")
	if err != nil {
		log.Fatal("Unable to create the file:", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	strRow := make([]string, len(result))
	for i, val := range result {
		strRow[i] = strconv.FormatFloat(val, 'f', 6, 64)
	}

	if err := writer.Write(strRow); err != nil {
		log.Fatal("Failed to write CSV", err)
	}

	log.Println("CSV file generated successfully！")

}

func LinearTrans(X [][]float64, k int, Slots int, ctVec []*rlwe.Ciphertext, params ckks.Parameters,
	ecd *ckks.Encoder, eval *ckks.Evaluator, kgen *rlwe.KeyGenerator, rlk *rlwe.RelinearizationKey,
	sk *rlwe.SecretKey) (lt lintrans.LinearTransformation, ltEval *lintrans.Evaluator) {

	diagsA := make([][]float64, k)
	for i := 0; i < k; i++ {
		diagsA[i] = make([]float64, k)
	}

	for i := 0; i < k; i++ {
		for j := 0; j < k; j++ {
			diagsA[j][i] = X[i][(i+j)%k]
		}
	}

	nonZeroDiagonals := make([]int, k)
	for i := 0; i < k; i++ {
		nonZeroDiagonals[i] = i
	}

	// We allocate the non-zero diagonals and populate them
	diagonals := make(lintrans.Diagonals[float64])

	for _, i := range nonZeroDiagonals {
		tmp := make([]float64, Slots)

		for j := 0; j < k; j++ {
			tmp[j] = diagsA[i][j]
		}

		diagonals[i] = tmp
	}

	ltparams := lintrans.Parameters{
		DiagonalsIndexList:        diagonals.DiagonalsIndexList(),
		LevelQ:                    ctVec[0].Level(),
		LevelP:                    params.MaxLevelP(),
		Scale:                     rlwe.NewScale(params.Q()[ctVec[0].Level()]),
		LogDimensions:             ctVec[0].LogDimensions,
		LogBabyStepGiantStepRatio: 1,
	}
	lt = lintrans.NewTransformation(params, ltparams)
	if err := lintrans.Encode(ecd, diagonals, lt); err != nil {
		panic(err)
	}

	galElsLt := lintrans.GaloisElements(params, ltparams)

	ltEval = lintrans.NewEvaluator(eval.WithKey(rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(galElsLt, sk)...)))
	return lt, ltEval
}

// Transpose returns the transpose of a 2D slice
func Transpose[T any](matrix [][]T) [][]T {
	if len(matrix) == 0 {
		return [][]T{}
	}
	rows := len(matrix)
	cols := len(matrix[0])
	result := make([][]T, cols)
	for i := range result {
		result[i] = make([]T, rows)
		for j := 0; j < rows; j++ {
			result[i][j] = matrix[j][i]
		}
	}
	return result
}

func isSquare(X [][]float64) (X_squ [][]float64) {
	rows := len(X)
	cols := 0
	if rows > 0 {
		cols = len(X[0])
	}

	size := rows
	if cols > size {
		size = cols
	}

	X_squ = make([][]float64, size)
	for i := range X_squ {
		X_squ[i] = make([]float64, size)
		for j := range X_squ[i] {
			if i < rows && j < cols {
				X_squ[i][j] = X[i][j]
			} else {
				X_squ[i][j] = 0 
			}
		}
	}

	return X_squ
}

func encTrain(X_train [][]float64, params ckks.Parameters, y_train []float64,
	enc *rlwe.Encryptor) (ctXt *rlwe.Ciphertext, cty *rlwe.Ciphertext) {

	var err error

	//var rowX []float64
	//
	//for _, row := range X_train {
	//	rowX = append(rowX, row...)
	//}
	//fmt.Println(rowX)

	//y := []float64{1.0, 0.0, 0.0, 1.0}
	//fmt.Println(y)

	//ptX := ckks.NewPlaintext(params, params.MaxLevel())
	pty := ckks.NewPlaintext(params, params.MaxLevel())

	// Encode
	ecd := ckks.NewEncoder(ckks.Parameters(params))
	//if err = ecd.Encode(rowX, ptX); err != nil {
	//	panic(err)
	//}

	if err = ecd.Encode(y_train, pty); err != nil {
		panic(err)
	}

	// Encrypt
	//ctX, err = enc.EncryptNew(ptX)
	//if err != nil {
	//	panic(err)
	//}

	cty, err = enc.EncryptNew(pty)
	if err != nil {
		panic(err)
	}

	Xt := Transpose(X_train)
	//fmt.Println("Xt:", Xt)
	var rowXt []float64

	for _, row := range Xt {
		rowXt = append(rowXt, row...)
	}

	ptXt := ckks.NewPlaintext(params, params.MaxLevel())
	if err = ecd.Encode(rowXt, ptXt); err != nil {
		panic(err)
	}
	ctXt, err = enc.EncryptNew(ptXt)
	if err != nil {
		panic(err)
	}

	return ctXt, cty
}

func SplitDataset(X [][]float64, y []float64, batchSize int) ([][][]float64, [][]float64) {
	n := len(X)
	X_batches := [][][]float64{}
	y_batches := [][]float64{}

	for i := 0; i < n; i += batchSize {
		end := i + batchSize
		if end > n {
			end = n
		}

		X_batch := X[i:end]
		y_batch := y[i:end]

		X_batches = append(X_batches, X_batch)
		y_batches = append(y_batches, y_batch)
	}

	return X_batches, y_batches
}
