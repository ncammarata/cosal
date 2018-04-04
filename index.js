const { readFileSync, writeFileSync } = require("fs")
const { flatten, uniq, sum, range } = require("lodash")
const { Matrix, pseudoInverse } = require("ml-matrix")
const cov = require("compute-covariance")

const embedding = {}
readFileSync("./glove.20k.txt", "utf8")
  .split("\n")
  .slice(0, 500)
  .forEach(line => {
    if (line.length === 0) {
      return
    }

    const [word, ...vectors] = line.split(" ")
    embedding[word] = vectors.map(i => +i)
  })

const dimensionality = Object.keys(embedding)[0].length

const getVector = word =>
  embedding[word] || range(dimensionality).map(i => Math.random())

const run = async () => {
  const text = readFileSync("./corpus.txt", "utf8")
  const lines = text
    .split("\n")
    .filter(i => i.length > 1)
    .map(line =>
      line
        .split(" ")
        .map(word => ({ word, vector: getVector(word) }))
        .filter(({ vector }) => vector.length > 1)
    )
  const pinv = m => new Matrix(m).pseudoInverse().to2DArray()

  const words = flatten(lines)
  const vecCov = vecs => cov(new Matrix(vecs).transpose().to2DArray())

  const covarianceDoc = new Matrix(vecCov(words.map(({ vector }) => vector)))
  const invCovDoc = pinv(covarianceDoc)
  const languageWords = Object.keys(embedding)
  const covarianceLanguage = new Matrix(
    vecCov(languageWords.slice(0, 100).map(word => embedding[word]))
  )

  const vectors = new Matrix(words.map(({ vector }) => vector))

  const p = Math.log(words.length) / Math.log(languageWords.length)
  const numerator = covarianceLanguage.add(p)
  const firstTerm = vectors.mmul(invCovDoc)
  const lastTerm = vectors.pseudoInverse()
  const inner = firstTerm.mmul(numerator.divide(p + 1)).mmul(lastTerm)
  const cosal = inner.sqrt()
  console.log("cosal is", cosal)
}

run()
