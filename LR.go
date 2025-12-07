package main

import (
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/lintrans"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func train(lt []lintrans.LinearTransformation, ltEval []*lintrans.Evaluator,
	k int, LogN int, ctVec0 *rlwe.Ciphertext,
	rotEval *ckks.Evaluator, rot int,
	rlk *rlwe.RelinearizationKey, kgen *rlwe.KeyGenerator, sk *rlwe.SecretKey,
	ctXt []*rlwe.Ciphertext, cty []*rlwe.Ciphertext,
	evalInnsum *ckks.Evaluator, batch int, Slots int,
	eval *ckks.Evaluator, params ckks.Parameters, enc *rlwe.Encryptor, btpEval *bootstrapping.Evaluator,
	rows int, K float64, batchNum int) (ct_theta *rlwe.Ciphertext, ct_bias *rlwe.Ciphertext) {

	theta := make([]float64, k)
	
	for i := range theta {
		theta[i] = 1
	}

	var err error
	pt_theta := ckks.NewPlaintext(params, params.MaxLevel())
	ecd := ckks.NewEncoder(ckks.Parameters(params))

	if err = ecd.Encode(theta, pt_theta); err != nil {
		panic(err)
	}

	ct_theta, err = enc.EncryptNew(pt_theta)
	if err != nil {
		panic(err)
	}

	bias := make([]float64, 1)
	pt_bias := ckks.NewPlaintext(params, params.MaxLevel())

	if err = ecd.Encode(bias, pt_bias); err != nil {
		panic(err)
	}

	ct_bias, err = enc.EncryptNew(pt_bias)
	if err != nil {
		panic(err)
	}

	rows1 := []float64{1.0 / float64(rows)}
	pt_rows1 := ckks.NewPlaintext(params, params.MaxLevel())

	if err = ecd.Encode(rows1, pt_rows1); err != nil {
		panic(err)
	}

	//losses := make([]float64, 0)

	alpha := 0.1

	iterations := 2

	for i := 0; i < iterations; i++ {
		for i := 0; i < batchNum-1; i++ {
			ct_Xtheta := HomomoMatMutiVec1(lt[i], ltEval[i], ct_theta, eval, k, LogN, ctVec0, rotEval, rot)

			ctz, err := eval.AddNew(ct_Xtheta, ct_bias)
			if err != nil {
				panic(err)
			}

			cth := sigmod(params, K, eval, ctz)
			
			maskVec := make([]float64, Slots)
			if k > Slots {
				k = Slots
			}
			for i := 0; i < k; i++ {
				maskVec[i] = 1
			}
			pt_maskVec := ckks.NewPlaintext(params, params.MaxLevel())

			if err = ecd.Encode(maskVec, pt_maskVec); err != nil {
				panic(err)
			}

			cth, err = eval.MulRelinNew(cth, pt_maskVec)
			if err != nil {
				panic(err)
			}

			if err = eval.Rescale(cth, cth); err != nil {
				panic(err)
			}

			ct_hsuby, err := eval.SubNew(cth, cty[i])
			if err != nil {
				panic(err)
			}

			ct_Xhy := HomomoMatMutiVec(rlk, kgen, sk, ctXt[i], ct_hsuby, params, Slots, eval, ecd, enc,
				k, evalInnsum, batch)

			ct_dtheta, err := eval.MulRelinNew(ct_Xhy, 1.0/float64(rows))
			if err != nil {
				panic(err)
			}

			if err = eval.Rescale(ct_dtheta, ct_dtheta); err != nil {
				panic(err)
			}

			if err := evalInnsum.InnerSum(ct_hsuby, batch, k, ct_hsuby); err != nil {
				panic(err)
			}
			ct_dbias, err := eval.MulRelinNew(ct_hsuby, pt_rows1)
			if err != nil {
				panic(err)
			}

			if err = eval.Rescale(ct_dbias, ct_dbias); err != nil {
				panic(err)
			}

			ct_dtheta, err = eval.MulRelinNew(ct_dtheta, alpha)
			if err != nil {
				panic(err)
			}

			if err = eval.Rescale(ct_dtheta, ct_dtheta); err != nil {
				panic(err)
			}

			ct_theta, err = eval.SubNew(ct_theta, ct_dtheta)
			if err != nil {
				panic(err)
			}

			ct_dbias, err = eval.MulRelinNew(ct_dbias, alpha)
			if err != nil {
				panic(err)
			}

			if err = eval.Rescale(ct_dbias, ct_dbias); err != nil {
				panic(err)
			}

			ct_bias, err = eval.SubNew(ct_bias, ct_dbias)
			if err != nil {
				panic(err)
			}

			ct_theta, err = btpEval.Bootstrap(ct_theta)
			if err != nil {
				panic(err)
			}
			ct_bias, err = btpEval.Bootstrap(ct_bias)
			if err != nil {
				panic(err)
			}
		}

	}

	return ct_theta, ct_bias
}

func predict(lt lintrans.LinearTransformation, ltEval *lintrans.Evaluator,
	k int, LogN int, ctVec0 *rlwe.Ciphertext,
	rotEval *ckks.Evaluator, rot int,
	ct_theta *rlwe.Ciphertext, ct_bias *rlwe.Ciphertext,
	eval *ckks.Evaluator, params ckks.Parameters, K float64) (ct_probs *rlwe.Ciphertext) {

	ct_Xtheta := HomomoMatMutiVec1(lt, ltEval, ct_theta, eval, k, LogN, ctVec0, rotEval, rot)

	ctz, err := eval.AddNew(ct_Xtheta, ct_bias)
	if err != nil {
		panic(err)
	}

	ct_probs = sigmod(params, K, eval, ctz)

	return ct_probs
}
