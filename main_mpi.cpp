// by Carlos De Bernardis and Ihab Al-Shaikhli
#include <mpi.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>

int mpi_error;

using namespace std;

// functor for getting sum of previous result and square of current element
template<typename T>
struct square
{
    T operator()(const T& Left, const T& Right) const
    {   
        return (Left + Right*Right);
    }
};

class Main
{
private:
    vector<string> args;
    const double golden_root = 1.27201964951407;
    double alpha, beta, gamma;
    int Smax;
    int m, n;
    int b;
    
    // MPI stuff
    int myRank;// = -1;
    int numProcs;// = -1;
    int root;// = 0;
    MPI_Comm Comm;// = MPI_COMM_NULL;
    
    void printUsage()
    {
        cerr << "Usage:" << endl
             << this->args[0] << " <inputfile> [outputfile]" << endl;
    }

    // generates a vector of m random elements ranging from 1 to b inclusive
    vector<int> randomColors()
    {
        vector<int> x(m);
        
        generate(x.begin(), x.end(), rand);
        
        for(vector<int>::iterator i = x.begin(); i != x.end(); ++i)
            *i = (*i % (b + 1)) + 1;
        
        return x;
    }
    
    // build W based on A and x
    vector< vector<int> > buildW(const vector< vector<int> > & A, const vector<int> & x)
    {
        vector< vector<int> > W(b+1, vector<int>(n, 0));
        for(int i = 0; i < m; ++i) {
            for(int j = 0; j < n; ++j) {
                W[x[i]-1][j] += A[i][j];
            }
        }
        
        return W;
    }
    
    // returns the updated cost based on W and sk, and updates the heuristic merit u
    double updateCost(const vector< vector<int> > & W, const vector<int> & sk, int &u)
    {
        vector<int> q(n, 0);
        
        for(int i = 0; i < b; ++i)
            for(int j = 0; j < n; ++j)
                if (W[i][j] > 0) q[j]++;
        
        int c = 0;
        for(vector<int>::iterator it = q.begin(); it != q.end(); ++it) 
            if (*it >= 2) c++;
        
        int r = sk[b];
        
        double hksum = 0.0;
        
        for(int i = 0; i < b; ++i)
            hksum += pow((sk[i]-(m/b)), 2);
        
        u = accumulate(q.begin(), q.end(), 0);
        
        return alpha*hksum + beta*c + gamma*r;
    }
    
    // performs the random genetic crossing of two solutions x, overwrites the second
    void cross(const int * x1, int * x2)
    {
        int oneortwo;
        
        for(int i = 0; i < m; ++i) {
            oneortwo = rand() % 2;
            x2[i] = x1[i]*oneortwo + x2[i]*(1 - oneortwo);
        }
    }
    
public:
    Main(vector<string> cmdlineargs) : args(cmdlineargs), 
                                       myRank(-1),
                                       numProcs(-1),
                                       root(0),
                                       Comm(MPI_COMM_NULL)
    {
    }
    
    int Run()
    {
        // the user didn't supply enough command line arguments
        if (args.size() < 2) {
            this->printUsage();
            return 1;
        }
        
        // try to open the input file
        string inFileName(args[1]);
        ifstream input(inFileName.c_str());
        if (!input) {
            cerr << "Error: Couldn't open file " << inFileName << endl;
            return 2;
        }
        
        cerr << "Working with file " << inFileName << endl;
        
        // optionally, the user may supply an output file
        // we try to open that, if we can't, we use stdout
        streambuf * buf;
        ofstream outputFile;
        if (args.size() > 2) {
            string outFileName(args[2]);
            outputFile.open(outFileName.c_str());
            if(!outputFile) {
                cerr << "Error: Couldn't open file " << outFileName << ", using stdout" << endl;
                buf = cout.rdbuf();
            } else {
                cerr << "Using output file " << outFileName << endl;
                buf = outputFile.rdbuf();
            }
        } else {
            buf = cout.rdbuf();
        }
        ostream output(buf);
        
        MPI_Comm_dup (MPI_COMM_WORLD, &Comm);
        
        mpi_error = MPI_Comm_rank (Comm, &myRank);
        mpi_error = MPI_Comm_size (Comm, &numProcs);

        fprintf(stderr, "I am %d of %d\n", myRank, numProcs);
        
        /**** HERE BEGINS THE ACTUAL ALGORITHM ****/
        int i, j;
        
        // read the parameters from the file
        input >> alpha >> beta >> gamma;
        input >> Smax; // TODO: instead of getting Smax from input, making it fixed at 200
        input >> m >> n;
        input >> b;
                
        // read the matrix A
        vector< vector<int> > A(m, vector<int>(n, 0));
        while (!!(input >> i >> j)) {
            A[i-1][j-1] = 1;
        }
        
        srand (time(NULL)*myRank);
        
        vector<int> bestx;
        
        // our initial color vector is random
        vector<int> x = randomColors(); // each MPI process will have a different x
        double cost;
        
        int iterations = 20;   //how many genetic iterations we wanna run
        Smax = Smax/8;
        // every step number of iterations we'll cool the system down
        int step = ceil(pow(log2(Smax) + log2(m) + log2(n), golden_root));
        // epsilon1 is the rate with which we cool the system down
        // epsilon2 is how we update mu
        double eps1 = 1.0/log(step);
        double eps2 = eps1 + 1.0/(m*n);
        int deaths = 1;
        
        while (iterations > 1) {
        
            // construct the matrix W to help compute the costs
            vector< vector<int> > W = buildW(A, x);
            
            // and sk too
            vector<int> sk(b+1, 0);
            for(vector<int>::iterator it = x.begin(); it != x.end(); ++it)
                sk[(*it)-1]++;
            
            // the updateCost function gives us the value of u, used to compute the heuristic cost
            int u; // heuristic merit
            cost = updateCost(W, sk, u);
            
            // let's print out an initial report to stderr
            fprintf(stderr, "%d's initial cost: %f\n", myRank, cost);
            
            // initialize mu and the inverse temperature with very low values
            double mu;
            double inverse_temp = mu = 1E-6;
            // number of accepted change
            int accepted = 0;
            // the heuristic cost
            double heuristic_cost = cost + u/mu;
            
            int s;
            // this is the annealing loop
            for(s=1; s<=Smax; s++) {
                if(s%step == 0) {
                    // we stop if we haven't accepted any changes and if the system is cool
                    if ((inverse_temp > 1) && (accepted < 1))
                        break;
                    // otherwise, we zero the accepted counter, cool the system down and continue    
                    accepted = 0;
                    inverse_temp += eps1*inverse_temp;
                    mu += eps2*mu;
                }
                
                // we choose a line at random and assign a new random color for it
                int rndline = rand() % m;
                int newcolor = (rand() % b) + 1;
                newcolor += (newcolor >= x[rndline]);
                
                for (j = 0; j < n; ++j) {
                    W[newcolor-1][j] += A[rndline][j];
                    W[x[rndline]-1][j] -= A[rndline][j];
                }
                
                sk[newcolor-1]++;
                sk[x[rndline]-1]--;
                
                double newcost = updateCost(W, sk, u);
                double newheuristic_cost = newcost + u/mu;
                double deltah = newheuristic_cost - heuristic_cost;
                double metropolis = exp(-inverse_temp * deltah);
                
                // if the cost improves or according to a metropolis probability, we accept
                if((deltah <= 0) || (rand() <= metropolis)) {
                    x[rndline] = newcolor;
                    cost = newcost;
                    heuristic_cost = newheuristic_cost;
                    accepted++;
                } else { // otherwise we reject
                    for (j = 0; j < n; ++j) {
                        W[newcolor-1][j] -= A[rndline][j];
                        W[x[rndline]-1][j] += A[rndline][j];
                    }
                    sk[newcolor-1]--;
                    sk[x[rndline]-1]++;
                }
            }
            
            // let's print out an initial report to stderr
            fprintf(stderr, "%d done annealing for now, best cost %f found in %d iterations\n", myRank, cost, s);
            
            int nselec = 0;
            int *rxbuf = NULL;
            double *rcostbuf = NULL;
            
            if (myRank == root) {
                rxbuf = new int[numProcs * m];
                rcostbuf = new double[numProcs];
            }
            
            MPI_Gather(&x.front(), m, MPI_INT, rxbuf, m, MPI_INT, root, Comm);
            MPI_Gather(&cost, 1, MPI_DOUBLE, rcostbuf, 1, MPI_DOUBLE, root, Comm);
            
            if (myRank == root) {
                vector<double> allcosts(rcostbuf, rcostbuf + numProcs);
                delete [] rcostbuf;
                
                double cost2All = accumulate(allcosts.begin(), allcosts.end(), 0, square<double>());
                double costAll = accumulate(allcosts.begin(), allcosts.end(), 0);
                double var_cost = (cost2All  - (costAll*costAll / numProcs ) ) / numProcs;
                
                nselec = floor(log10(var_cost));
                if (nselec > numProcs/2) nselec = numProcs/2;
                if (nselec > 10) nselec = 10;
                
                vector< pair<double, int> > vp;
                vp.reserve(allcosts.size());
                for (i = 0; i < allcosts.size(); ++i) {
                    vp.push_back(make_pair(allcosts[i], i));
                }
                
                sort(vp.begin(), vp.end());
                
                // also, let's print a periodical report to stderr
                cost = vp[0].first;
                bestx.assign(&(rxbuf[vp[0].second*m]), &(rxbuf[vp[0].second*m + m]));
                fprintf(stderr, "\n%f is the best cost found thus far, with a variance of %f; doing %d crossings\n\n", vp[0].first, var_cost, nselec);
                
                for (i = 0; i < nselec; ++i)
                    cross(&(rxbuf[vp[i].second*m]), &(rxbuf[vp[numProcs-i-1].second*m]));
            }
            
            int *r2xbuf = new int[m];
            MPI_Scatter(rxbuf, m, MPI_INT, r2xbuf, m, MPI_INT, root, Comm);
            
            if (myRank == root) {
                delete [] rxbuf;
            }
            
            x.assign(r2xbuf, r2xbuf + m);
            
            delete [] r2xbuf;
            
            if (myRank == root) {
                iterations += nselec;
                iterations -= deaths++;
            }
            
            MPI_Bcast(&iterations, 1, MPI_INT, root, Comm);
        }
        
        if (myRank == root) {
            output << "Final solution's cost: " << cost << endl;
            output << "With solution vector x = [ ";
            for(vector<int>::iterator it = bestx.begin(); it != bestx.end(); ++it)
                output << *it << " ";
            output << "]" << endl;
        }
        
        MPI_Finalize();
        return 0;
    }
    
};

int main(int argc, char *argv[])
{
    // Initializing MPI
    mpi_error = MPI_Init(&argc, &argv);
        
    vector<string> args(argv, argv + argc);
    
    Main app(args);
    
	return app.Run();
}
