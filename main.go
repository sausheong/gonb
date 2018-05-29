package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"time"
)

// parameters
var (
	testPercentage = 0.1
	datafile       = "amazon.txt"
	threshold      = 1.1
)

// datasets
type document struct {
	sentiment string
	text      string
}

var train []document
var test []document

var categories = []string{"1", "0"}

func main() {
	setupData(datafile)
	fmt.Println("Data file used:", datafile)
	fmt.Println("no of docs in TRAIN dataset:", len(train))
	fmt.Println("no of docs in TEST dataset:", len(test))

	c := createClassifier(categories, threshold)

	// train on train dataset
	for _, doc := range train {
		c.Train(doc.sentiment, doc.text)
	}

	// validate on test dataset
	count, accurates, unknowns := 0, 0, 0
	for _, doc := range test {
		count++
		sentiment := c.Classify(doc.text)
		if sentiment == doc.sentiment {
			accurates++
		}
		if sentiment == "unknown" {
			unknowns++
		}
	}
	fmt.Printf("Accuracy on TEST dataset is %2.1f%% with %2.1f%% unknowns", float64(accurates)*100/float64(count), float64(unknowns)*100/float64(count))

	// validate on the first 100 docs in the train dataset
	count, accurates, unknowns = 0, 0, 0
	for _, doc := range train[0:100] {
		count++
		sentiment := c.Classify(doc.text)
		if sentiment == doc.sentiment {
			accurates++
		}
		if sentiment == "unknown" {
			unknowns++
		}
	}
	fmt.Printf("\nAccuracy on TRAIN dataset is %2.1f%% with %2.1f%% unknowns", float64(accurates)*100/float64(count), float64(unknowns)*100/float64(count))

}

// set up data for training and testing
func setupData(file string) {
	rand.Seed(time.Now().UTC().UnixNano())
	data, err := readLines(file)
	if err != nil {
		fmt.Println("Cannot read file", err)
		os.Exit(1)
	}
	for _, line := range data {
		s := strings.Split(line, "|")
		doc, sentiment := s[0], s[1]

		if rand.Float64() > testPercentage {
			train = append(train, document{sentiment, doc})
		} else {
			test = append(test, document{sentiment, doc})
		}
	}
}

// read the file line by line
func readLines(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	return lines, scanner.Err()
}
