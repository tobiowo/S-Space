/*
 * Copyright 2010 Keith Stevens 
 *
 * This file is part of the S-Space package and is covered under the terms and
 * conditions therein.
 *
 * The S-Space package is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as published
 * by the Free Software Foundation and distributed hereunder to you.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" AND NO REPRESENTATIONS OR WARRANTIES,
 * EXPRESS OR IMPLIED ARE MADE.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, WE MAKE
 * NO REPRESENTATIONS OR WARRANTIES OF MERCHANT- ABILITY OR FITNESS FOR ANY
 * PARTICULAR PURPOSE OR THAT THE USE OF THE LICENSED SOFTWARE OR DOCUMENTATION
 * WILL NOT INFRINGE ANY THIRD PARTY PATENTS, COPYRIGHTS, TRADEMARKS OR OTHER
 * RIGHTS.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

package edu.ucla.sspace.clustering;

import edu.ucla.sspace.matrix.Matrix;
import edu.ucla.sspace.matrix.Matrices;

import edu.ucla.sspace.vector.DoubleVector;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;


/**
 * An implementation of the Bisecting K-Means algorithm, also known as Repeated
 * Bisections.  This implementation is based on the following paper:
 *
 *   <li style="font-family:Garamond, Georgia, serif">Michael Steinbach,
 *   George Karypis, Vipin Kumar.  "A comparison of document clustering
 *   techniques," in <i>KDD Workshop on Text Mining</i>, 2000</li>
 *
 * This clustering algorithm improves upon the standard K-Means algorithm by
 * taking a data set and repeatedly splitting the data points into two regions.
 * Initially all data points are separated into two clusters.  Then, until the
 * desired number of clusters are created, the largest cluster is divided using
 * K-Means with K equal to 2.  This implementation relies on the {@link
 * DirectClustering} implementation.  Any properties passed to this clustering
 * method are passed onto the {@link DirectClustering} algorithm, allowing the
 * user to set the desired seeding method.
 *
 * @see KMeansClustering
 * 
 * @author Keith Stevens
 */
public class BisectingKMeans implements Clustering {
    public static double cost0;
    public static double cost1;
    public static double[] costs;
    private static ArrayList<ArrayList<Integer>> clusterFreqs;
    public static ArrayList<Integer> splitIndex;
    public static int[] numAssignments;
    public static double splitAssignment;
    /**
     * Not implemented.
     */
    public Assignments cluster(Matrix dataPoints, Properties props) {
        throw new UnsupportedOperationException(
                "KMeansClustering requires that the " +
                "number of clusters be specified");
    }

    /**
     * {@inheritDoc}
     */
    public Assignments cluster(Matrix dataPoints,
                               int numClusters,
                               Properties props) {
        // Handle a simple base case.
        if (numClusters <= 1) {
            Assignment[] assignments = new Assignment[dataPoints.rows()];
            for (int i = 0; i < assignments.length; ++i)
                assignments[i] = new HardAssignment(0);
            return new Assignments(numClusters, assignments, dataPoints);
        }

        // Create a count of cluster assignments.
        numAssignments = new int[numClusters];
        costs = new double[numClusters];
        
        // Create a list of lists.  The inner list represents the vectors
        // assigned to a particular cluster.  We use this method so that we can
        // easily transform the cluster to a Matrix
        List<List<DoubleVector>> clusters = new ArrayList<List<DoubleVector>>(
                numClusters);
        // Keeps track of the frequencies of each token in each cluster
        clusterFreqs = new ArrayList<ArrayList<Integer>>(numClusters);
        for (int c = 0; c < numClusters; ++c) {
            clusters.add(new ArrayList<DoubleVector>());
            clusterFreqs.add(new ArrayList<Integer>());
        }
        splitIndex = clusterFreqs.get(0);
        for (int i = 0; i < dataPoints.rows(); i++) {
            splitIndex.add(LDAkmeans.freqs[i]);
            numAssignments[0] += LDAkmeans.freqs[i];
        }
        splitAssignment = numAssignments[0];
        List<Integer> newIndex = clusterFreqs.get(1); 
        
        Clustering clustering = new DirectClustering();
        // Make the first bisection.
        Assignment[] assignments =
            clustering.cluster(dataPoints, 2, props).assignments();
        costs[0] = cost0;
        costs[1] = cost1;
        numAssignments[0] = 0;
        // Count the number of assignments made to each cluster and move the
        // vectors in to the corresponding list.
        for (int i = 0; i < assignments.length; ++i) {
            int assignment = assignments[i].assignments()[0];
            LDAkmeans.paths[i] = new StringBuilder(12);
            LDAkmeans.paths[i].append(assignment);
            numAssignments[assignment]+= LDAkmeans.freqs[i];
            clusters.get(assignment).add(dataPoints.getRowVector(i));
            clusterFreqs.get(assignment).add(LDAkmeans.freqs[i]);
        }
        System.err.printf("Split #1: into %,d and %,d types; %,d and %,d tokens\t",
            clusters.get(0).size(), clusters.get(1).size(),
            numAssignments[0], numAssignments[1]);
        System.err.printf("s1:%.2f,s2:%.2f\n", costs[0], costs[1]);
        // Generate the numClusters - 2 clusters by finding the largest cluster
        // and bisecting it.  Of the 2 resulting clusters, one will maintain the
        // same cluster index and the other will be given a new cluster index,
        // namely k, the current cluster number.
        for (int k = 2; k < numClusters; k++) {
/*            // Find the largest cluster (tokens).
            //only works when you have the corpus frequencies
            int largestSize = 0;
            int largestIndex = 0;
            for (int c = 0; c < k; ++c) {
                int size = numAssignments[c];
                //TODO: choose a good split size threshold 
                if (size > largestSize && clusters.get(c).size() > 10) {
                    largestSize = size;
                    largestIndex = c;
                }
            }*/

/*            // Find the largest cluster (types).
            int largestSize = 0;
            int largestIndex = 0;
            for (int c = 0; c < k; ++c) {
                int size = clusters.get(c).size();
                //TODO: choose a good split size threshold 
                if (size > largestSize && clusters.get(c).size() > 10) {
                    largestSize = size;
                    largestIndex = c;
                }
            }*/

            //Find the lowest scoring cluster based on criterion
            double lowestScore = Double.MAX_VALUE;
            int largestIndex = 0;
            for (int c = 0; c < k; ++c) {
                double score = costs[c];
                if (score < lowestScore && clusters.get(c).size() > 10) {
                    lowestScore = score;
                    largestIndex = c;
                }
            }

            if (clusters.get(largestIndex).size() == 0) {
                System.err.printf("Done clustering at k=%d\n", k);
                break;
            }

            // Get the list of vectors representing the cluster being split and
            // the cluster that will hold the vectors split off from this
            // cluster.
            List<DoubleVector> originalCluster = clusters.get(largestIndex);
            List<DoubleVector> newCluster = clusters.get(k);
            splitIndex = clusterFreqs.get(largestIndex);
            newIndex = clusterFreqs.get(k);
            splitAssignment = numAssignments[largestIndex];
            // Split the largest cluster.
            System.err.printf("splitting cluster with %,d types, %,d tokens, s=%f...\n",
                clusters.get(largestIndex).size(), numAssignments[largestIndex], costs[largestIndex]);
            Matrix clusterToSplit = Matrices.asMatrix(originalCluster);
            
            Assignment[] newAssignments = 
                clustering.cluster(clusterToSplit, 2, props).assignments();
            costs[largestIndex] = cost0;
            costs[k] = cost1;
            // Clear the lists for cluster being split and the new cluster.
            // Also clear the number of assignments.
            originalCluster.clear();
            newCluster.clear();
            splitIndex.clear();
            numAssignments[largestIndex] = 0;
            numAssignments[k] = 0;

            // Reassign data points in the largest cluster.  Data points
            // assigned to the 0 cluster maintain their cluster number in the
            // real assignment list.  Data points assigned to cluster 1 get the
            // new cluster number, k.  
            for (int i = 0, j = 0; i < dataPoints.rows(); ++i) {
                if (assignments[i].assignments()[0] == largestIndex) {
                    // Make the assignment for vectors that keep their
                    // assignment.
                    if (newAssignments[j].assignments()[0] == 0) {
                        originalCluster.add(dataPoints.getRowVector(i));
                        numAssignments[largestIndex]+= LDAkmeans.freqs[i];
                        splitIndex.add(LDAkmeans.freqs[i]);
                        LDAkmeans.paths[i].append(0);
                    }
                    // Make the assignment for vectors that have changed their
                    // assignment.
                    else {
                        newCluster.add(dataPoints.getRowVector(i));
                        assignments[i] = new HardAssignment(k);
                        numAssignments[k]+= LDAkmeans.freqs[i];
                        newIndex.add(LDAkmeans.freqs[i]);
                        LDAkmeans.paths[i].append(1);
                    }
                    j++;
                }
            }
            if (clusters.get(largestIndex).isEmpty() || clusters.get(k).isEmpty()) {
                //don't try to split clusters again if they failed to split
                //even though it was probably because of bad seeding
                costs[largestIndex] = Double.MAX_VALUE;
                costs[k] = Double.MAX_VALUE;
            }
            System.err.printf("Split #%d: into %,d and %,d types; %,d and %,d tokens\t",
                k, clusters.get(largestIndex).size(), clusters.get(k).size(),
                numAssignments[largestIndex], numAssignments[k]);
            System.err.printf("s1:%.2f,s2:%.2f\n", costs[largestIndex], costs[k]);
        }
        return new Assignments(numClusters, assignments, dataPoints);
    }

    public String toString() {
        return "BisectingKMeans";
    }
}
