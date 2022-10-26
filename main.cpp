#include <iostream>
#include <random>
#include <mpi.h>
#include <omp.h>

static const double predefined_res = 1./24;
static double eps;
static long points_batch_num = 10000;

struct Point
{
    double x,y,z;
};

#define return_finalize(exit_num) MPI_Finalize(); \
                                  return (exit_num);

double f( double x, double y, double z)
{
    return x*x*x*y*y*z;
}

bool isInArea( double x, double y, double z)
{
    return -1 <= x && x <= 0
        && -1 <= y && y <= 0
        && -1 <= z && z <= 0;
}

double F( double x, double y, double z) { return isInArea( x, y, z) ? f( x, y, z) : 0; }

class RandomGenerator
{
  public:
    RandomGenerator( double lower_bound, double upper_bound, uint32_t seed = 0)
        : unif( lower_bound, upper_bound)
    {
        RE.seed(seed);
    }

    double rand() { return unif( RE); }
  private:
    std::uniform_real_distribution<double> unif;
    std::default_random_engine RE;
};

/**
 * Entry point.
 * Input parametrs: ./monte-carlo <EPS>
 */
int main( int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);

    if( argc != 2 )
    {
        std::cout << "WRONG ARG NUMBER!" << std::endl;
        std::cout << "Но ладно, так уж и быть я представлю, что эпсилон = 10e-5...\n";
        eps = 10e-5;
        // return_finalize(-1);
    } else
    {
        eps = std::atof(argv[1]);
    }

    long local_points_num = 0;
    long total_points = 0;
    double local_time = 0;
    double res = 0;
    points_batch_num = (points_batch_num - 1) / size + 1;

    local_time = -MPI_Wtime();

#if not defined(MASTER) and not defined(DISTRIBUTED)
    static_assert( 0, "\nChoose type of communications via compilation flags:\n -DMASTER      for Master-Slave model.\n -DDISTRIBUTED for distributed model.\n");
#endif

double sum = 0;
int batchs_num = 0;
bool do_continue;
#if defined(MASTER)
    if ( rank == 0 )
    {
        RandomGenerator r( 0, 1,  137 + rank);
        Point points[(size-1)*points_batch_num];
        // Point *points = new Point[(size-1)*points_batch_num];
        do
        {
            batchs_num++;
            do_continue = false;
            #pragma omp parallel for
            for ( int i = 1; i < size; ++i )
            {
                for ( long j = 0; j < points_batch_num; ++j )
                {
                    points[(i-1)*points_batch_num + j].x = r.rand();
                    points[(i-1)*points_batch_num + j].y = r.rand();
                    points[(i-1)*points_batch_num + j].z = r.rand();
                }
                MPI_Request rq;
                MPI_Isend( (char *)(points+(i-1)*points_batch_num), sizeof(Point)*points_batch_num, MPI_CHAR, i, 0, MPI_COMM_WORLD, &rq);
            }
            MPI_Reduce( MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            // Calc
            res = (double)sum / (batchs_num * (size-1) * points_batch_num);
            std::cout << res << std::endl;
            do_continue = std::abs(predefined_res - res) > eps;
            MPI_Bcast( &do_continue, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        } while ( do_continue );
        total_points = batchs_num*points_batch_num*(size - 1);
    } else
    {
        Point points[points_batch_num];
        do
        {
            sum = 0;
            do_continue = false;
            MPI_Recv( (char*)points, sizeof(Point)*points_batch_num, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for ( int i = 0; i < points_batch_num; ++i )
            {
                sum += F( points[i].x, points[i].y, points[i].z);
            }
            MPI_Reduce( &sum, nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Bcast( &do_continue, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        } while ( do_continue );
    }

#elif defined(DISTRIBUTED)
    RandomGenerator r( -1, 0, 137 + rank);
    do
    {
        batchs_num++;
        do_continue = false;
        for ( long j = 0; j < points_batch_num; ++j )
        {
            double x = r.rand();
            double y = r.rand();
            double z = r.rand();
            sum += F( x, y, z);
        }

        double total_sum;
        MPI_Reduce( &sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if ( rank == 0 )
        {
            res = (double)total_sum / (batchs_num * size * points_batch_num);
            // std::cout << std::abs(predefined_res - res) << " " << eps << " " << res << std::endl;
            do_continue = std::abs(predefined_res - res) > eps;
        }
        MPI_Bcast( &do_continue, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    } while ( do_continue );
    total_points = batchs_num*points_batch_num*size;
#else
    return_finalize(-1);
#endif
    local_time += MPI_Wtime();
    double max_time;
    MPI_Reduce( &local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if( rank == 0 )
    {
        double error = std::abs(res - predefined_res);
        std::cout << res << std::endl;
        std::cout << error << std::endl;
        std::cout << total_points << std::endl;
        std::cout << max_time << std::endl;
    }

    return_finalize(0);
}
