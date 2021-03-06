/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */

/* PMSIS includes. */
#include "pmsis.h"

/* Autotiler includes. */
#include "Gap.h"
#include "Cifar10Kernels.h"
#include "CifarData.h"

/* Variables used. */
#define COEF_L2
#define ITERATIONS 1000
#define PERF
/* Board Setup. */
#define FREQ_FC (150*1000000)
#define FREQ_CL (90*1000000)
#define ALIM_1_VOLT 0
//#define DEBUG

#ifdef DEBUG
#define DEBUG_PRINTF printf
#else
#define DEBUG_PRINTF(...) ((void) 0)
#endif

#ifdef COEF_L2
#include "coef/CifarCoeff.h"
#else
#include "bsp/fs.h"
#include "bsp/flash/hyperflash.h"

#define  BUFFER_SIZE   (1024)

struct fs_conf conf_fs;
struct hyperflash_conf conf_flash;
struct hyperram_conf conf_ram;
struct pi_device flash;
struct pi_device fs;
struct pi_device ram;
fs_file_t *file;
int16_t *buff;
#endif  /* COEF_L2 */

uint16_t *Filter_Layer[3] = {0};
uint16_t *Bias_Layer[3]= {0};
uint32_t Filter_Layer_Size[3] = {0};
uint32_t Bias_Layer_Size[3]= {0};
int16_t *Out_Layer[3];
uint32_t Out_Layer_Size[3] = {0};

int ConvAt(short *In, short int *Filter, unsigned int X, unsigned int Y, unsigned int W, unsigned int H, unsigned int Norm)
{
    int i, j;
    int Acc = 0;
    int K = 5;

    for (i=0; i<K; i++) {
        for (j=0; j<K; j++) {
            Acc += In[(X+i)*W+Y+j]*Filter[K*i+j];
        }
    }
    return (gap_clip(gap_roundnorm_reg(Acc, Norm), 15));
}


void DumpPlane(char *Mess, short int *Plane, unsigned int W, unsigned int H)
{
    unsigned int i, j;

    printf("----------------- %s ------------------------\n", Mess);
    for (i=0; i<H; i++) {
        printf("%2d: ", i);
        for (j=0; j<W; j++) {
            printf("%4x ", (unsigned short) Plane[i*W+j]);
        }
        printf("\n");
    }
    printf("-----------------------------------------\n");
}

void DumpPaddedCoeff(char *Name, short int *C, unsigned int NTap, unsigned int NFilter)
{
    unsigned int i, j;
    printf("L2_MEM short int %s[] = {\n", Name);
    for (i=0; i<NFilter; i++) {
        for (j=0; j<NTap; j++) {
            printf("%d, ", C[i*NTap+j]);
        }
        printf("0,\n");
    }
    printf("};\n");
}

unsigned int CheckSum(short int *In, int Size)
{
    int i;
    unsigned S=0;

    for (i=0; i<Size; i++) S += In[i];
    return S;
}

void Check(char *Mess, short int *Planes, int NPlane, int W, int H)
{
    int i;

    printf("Check sum for %s\n", Mess);

    for (i=0; i<NPlane; i++) {
        printf("\t%2d: %x\n", i, CheckSum(Planes + i*(W*H), W*H));
    }
}

#if !defined(COEF_L2)
void CopyFileFromFlashToL3(struct pi_device *fs, char *file_name, uint32_t *hyper_buff, uint32_t *hyper_size)
{
    DEBUG_PRINTF("Loading layer \"%s\" from FS to L3\n", file_name);
    file = fs_open(fs, file_name, 0);
    if (file == NULL)
    {
        printf("File open failed !\n");
        pmsis_exit(-1);
    }

    if (ram_alloc(&ram, hyper_buff, file->size))
    {
        printf("Ram malloc failed !\n");
        pmsis_exit(-2);
    }
    *hyper_size = file->size;
    DEBUG_PRINTF("Hyperram alloc : %x %d\n", *hyper_buff, file->size);

    uint32_t size_total = 0;
    uint32_t size = 0;
    pi_task_t task;
    do
    {
        size = fs_read_async(file, buff, BUFFER_SIZE, pi_task_block(&task));
        pi_task_wait_on(&task);
        size = ((size + 3) & ~3);
        if (size)
        {
            ram_write(&ram, (uint32_t) (*hyper_buff+size_total), buff, size);
            size_total += size;
        }
    } while (size_total < file->size);
    DEBUG_PRINTF("Loading layer \"%s\" from FS to L3, hyper %x size = %d\n", file_name, *hyper_buff, size_total);

    fs_seek(file, 0);
    fs_close(file);
}
#endif  /* COEF_L2 */

static void RunCifar10(void *arg)
{
	int start_time;
	int checkpoint1;
	int checkpoint2;
	int checkpoint3;
	int stage1_time;
	int stage2_time;
	int stage3_time;
    DEBUG_PRINTF("Cluster: Start to run Cifar10\n");

    //rt_perf_conf(perf, (1<<perf_cnt_mode));
    //rt_perf_reset(perf);
    //rt_perf_start(perf);
	start_time = rt_time_get_us();
	printf("start_time %d\n", start_time);

    Conv5x5MaxPool2x2_SW_0(ImageIn,
                           Filter_Layer[0],
                           Bias_Layer[0],
                           Out_Layer[0],
                           14);
	checkpoint1 = rt_time_get_us();
	stage1_time = checkpoint1 - start_time;
	printf("%d\n", stage1_time);

    //perf_cnt = pi_perf_read(perf_cnt_mode);    
    //rt_perf_stop(perf);
    //printf("Counters: %d %s\n",perf_cnt,perf_cnt_name);
    //rt_perf_reset(perf);
    //rt_perf_start(perf);

    Conv5x5MaxPool2x2_SW_1(Out_Layer[0],
                           Filter_Layer[1],
                           Bias_Layer[1],
                           Out_Layer[1],
                           14);
	
	checkpoint2 = rt_time_get_us();
	stage2_time = checkpoint2 - checkpoint1;
	printf("%d\n", stage2_time);
	//perf_cnt = pi_perf_read(perf_cnt_mode);    
    //rt_perf_stop(perf);
    //printf("Counters: %d %s\n",perf_cnt,perf_cnt_name);
    //rt_perf_reset(perf);
    //rt_perf_start(perf);

    LinearLayerReLU_1(Out_Layer[1],
                      Filter_Layer[2],
                      Bias_Layer[2],
                      Out_Layer[2],
                      16,
                      10);

    //perf_cnt = pi_perf_read(perf_cnt_mode);    
    //rt_perf_stop(perf);
    //printf("Counters: %d %s\n",perf_cnt,perf_cnt_name);
	
	checkpoint3 = rt_time_get_us();
	stage1_time = checkpoint3 - checkpoint2;
	printf("%d\n", stage3_time);
    DEBUG_PRINTF("Cluster: End run Cifar10\n");
}

void test_cifar10(void)
{
    //printf("Entering main controller\n");
    uint8_t CheckResults = 1;

    /* Output result size. */
    Out_Layer_Size[0] = (14 * 14 * sizeof(int16_t) * 8);
    Out_Layer_Size[1] = (5 * 5 * sizeof(int16_t) * 12);
    Out_Layer_Size[2] = (1 * 1 * sizeof(int16_t) * 10);

    #if !defined(COEF_L2)
    buff = (int16_t *) pmsis_l2_malloc(BUFFER_SIZE);
    if (buff == NULL)
    {
        printf("Buffer alloc failed !\n");
        pmsis_exit(-1);
    }

    hyperram_conf_init(&conf_ram);
    pi_open_from_conf(&ram, &conf_ram);
    if (ram_open(&ram))
    {
        printf("Error ram open !\n");
        pmsis_exit(-2);
    }

    hyperflash_conf_init(&conf_flash);
    pi_open_from_conf(&flash, &conf_flash);
    if (flash_open(&flash))
    {
        printf("Error flash open !\n");
        pmsis_exit(-3);
    }

    fs_conf_init(&conf_fs);
    conf_fs.flash = &flash;
    pi_open_from_conf(&fs, &conf_fs);

    int32_t err = fs_mount(&fs);
    if (err)
    {
        printf("Error FS mounting : %d !\n", err);
        pmsis_exit(-4);
    }
    printf("FS mounted.\n");

    CopyFileFromFlashToL3(&fs, "Cifar10_Filter0.dat", &Filter_Layer[0], &Filter_Layer_Size[0]);
    CopyFileFromFlashToL3(&fs, "Cifar10_Bias0.dat",   &Bias_Layer[0], &Bias_Layer_Size[0]);
    CopyFileFromFlashToL3(&fs, "Cifar10_Filter1.dat", &Filter_Layer[1], &Filter_Layer_Size[1]);
    CopyFileFromFlashToL3(&fs, "Cifar10_Bias1.dat",   &Bias_Layer[1], &Bias_Layer_Size[1]);
    CopyFileFromFlashToL3(&fs, "Cifar10_Filter2.dat", &Filter_Layer[2], &Filter_Layer_Size[2]);
    CopyFileFromFlashToL3(&fs, "Cifar10_Bias2.dat",   &Bias_Layer[2], &Bias_Layer_Size[2]);

    fs_unmount(&fs);
    printf("FS unmounted.\n");
    flash_close(&flash);

    if (ram_alloc(&ram, (uint32_t *) &Out_Layer[0], Out_Layer_Size[0]))
    {
        printf("Ram malloc failed !\n");
        pmsis_exit(-5);
    }
    #else
    /* Bias & Filters. */
    Bias_Layer[0] = Bias_Layer0;
    Bias_Layer[1] = Bias_Layer1;
    Bias_Layer[2] = Bias_Layer2;
    Filter_Layer[0] = Filter_Layer0;
    Filter_Layer[1] = Filter_Layer1;
    Filter_Layer[2] = Filter_Layer2;

    Out_Layer[0] = (int16_t *) pmsis_l2_malloc(Out_Layer_Size[0]);
    if (Out_Layer[0] == NULL)
    {
        printf("Failed to allocated memory, giving up.\n");
        pmsis_exit(-5);
    }
    #endif  /* COEF_L2 */
    else
    {
        //printf("Allocating %d: OK -> %x\n", Out_Layer_Size[0], Out_Layer[0]);
    }

    Out_Layer[1] = (int16_t *) pmsis_l2_malloc(Out_Layer_Size[1]);
    Out_Layer[2] = (int16_t *) pmsis_l2_malloc(Out_Layer_Size[2]);
    if ((Out_Layer[1] == NULL) && (Out_Layer[2] == NULL))
    {
        printf("Failed to allocated memory, giving up.\n");
        pmsis_exit(-5);
    }
    else
    {
        //printf("Allocating %d: OK -> %x\n", Out_Layer_Size[1], Out_Layer[1]);
        //printf("Allocating %d: OK -> %x\n", Out_Layer_Size[2], Out_Layer[2]);
    }

    /* Configure And open cluster. */
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    cl_conf.id = 0;
    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-7);
    }

    /* Allocate the predetermined memory in the shared L1 memory that the cluster can act on. */
    Cifar10_L1_Memory = pmsis_l1_malloc(_Cifar10_L1_Memory_SIZE);
    if (Cifar10_L1_Memory == NULL)
    {
        printf("Memory Allocation Error! Quit...\n");
        pmsis_exit(-8);
    }

    struct pi_cluster_task *task = pmsis_l2_malloc(sizeof(struct pi_cluster_task));
    memset(task, 0, sizeof(struct pi_cluster_task));
    task->entry = RunCifar10;
    task->arg = NULL;
    task->stack_size = 2048*2;

    pi_cluster_send_task_to_cl(&cluster_dev, task);

    pmsis_l1_malloc_free(Cifar10_L1_Memory, _Cifar10_L1_Memory_SIZE);

    pi_cluster_close(&cluster_dev);

   /* if (CheckResults)
    {
        printf("L1: ");
        Check("SW   Layer0", Out_Layer[0], 8, 14, 14);
        printf("L2: ");
        Check("SW   Layer1", Out_Layer[1], 12, 5, 5);
        printf("L3: ");
        Check("SW   Layer2", Out_Layer[2], 10, 1, 1);
    }*/

    #if !defined(COEF_L2)
    for (uint32_t i = 0; i < 3; i++)
    {
        ram_free(&ram, Filter_Layer[i], Filter_Layer_Size[i]);
        ram_free(&ram, Bias_Layer[i], Bias_Layer_Size[i]);
    }
    ram_free(&ram, (uint32_t) Out_Layer[0], Out_Layer_Size[0]);
    ram_close(&ram);
    pmsis_l2_malloc_free(buff, BUFFER_SIZE);
    #else
    pmsis_l2_malloc_free(Out_Layer[0], Out_Layer_Size[0]);
    #endif  /* COEF_L2 */
    pmsis_l2_malloc_free(Out_Layer[1], Out_Layer_Size[1]);
    pmsis_l2_malloc_free(Out_Layer[2], Out_Layer_Size[2]);


    //printf("Test success\n");
    //pmsis_exit(0);
}

int main(void)
{
	//Set Fabric Controller and Cluster Frequencies
    rt_freq_set(RT_FREQ_DOMAIN_FC, FREQ_FC);
    rt_freq_set(RT_FREQ_DOMAIN_CL, FREQ_CL);
	#if !ALIM_1_VOLT
    PMU_set_voltage(1150,0);
    PMU_set_voltage(1200,0);
    #else
    PMU_set_voltage(1000,0);
    #endif
	#ifdef PERF
	long int start_time, end_time;
	long int tot_time;
	unsigned int Ti;
	rt_perf_t *perf;
	perf = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(rt_perf_t));
	rt_perf_init(perf);
	int perf_cnt = 0; //change this for different counters
	int perf_cnt_tot = 0;
	int perf_cnt_mode;
	char* perf_cnt_name;
	switch(perf_cnt){
		case 0:
			perf_cnt_mode = RT_PERF_CYCLES;
			perf_cnt_name = "RT_PERF_CYCLES";
			break;
		case 1:
			perf_cnt_mode = RT_PERF_ACTIVE_CYCLES;
			perf_cnt_name = "RT_PERF_ACTIVE_CYCLES";
			break;
		case 2:
			perf_cnt_mode = RT_PERF_INSTR;
			perf_cnt_name = "RT_PERF_INSTR";
			break;
		case 3:
			perf_cnt_mode = RT_PERF_IMISS;
			perf_cnt_name = "RT_PERF_IMISS";
			break;
		case 4:
			perf_cnt_mode = RT_PERF_LD;
			perf_cnt_name = "RT_PERF_LD";
			break;
		case 5:
			perf_cnt_mode = RT_PERF_ST;
			perf_cnt_name = "RT_PERF_ST";
			break;
	}	
	rt_perf_conf(perf, ((1<<RT_PERF_CYCLES)|(1<<perf_cnt_mode)));
	rt_perf_reset(perf);
	rt_perf_start(perf);
	start_time = rt_time_get_us();
	printf("\n\n\t *** PMSIS Cifar10 Test ***\n\n");
	for (int j; j<ITERATIONS; j++){
		 pmsis_kickoff((void *) test_cifar10);
		 printf("%d\n",j);
	}
	rt_perf_stop(perf);
	perf_cnt_tot = pi_perf_read(RT_PERF_CYCLES);
	perf_cnt = pi_perf_read(perf_cnt_mode);
    end_time = rt_time_get_us();	
	
	tot_time = end_time - start_time;
	printf("Time: %d uSec. | Counters: %d %s\n",tot_time,perf_cnt,perf_cnt_name);
	#else
	for (int j; j<ITERATIONS; j++){
		 pmsis_kickoff((void *) test_cifar10);
		 printf("%d\n",j);
	}
	#endif
	printf("Test complete");
}

