const { matrix, inv } = require("mathjs")
const { range } = require("lodash")
const cov = require("compute-covariance")

x = range(10).map(x => range(100).map(y => Math.random()))
console.log(inv(matrix(cov(x))))
