from pathlib import Path
from util.utility import MakeFolderWithCurrentFuncName
from chemistry.visualization import WriteDataFrameSmilesToXls
import pandas as pd
from fastsolv import fastsolv

def RunPredSolv(df, solute_smiles_col, solvent_smiles, temp):
    # wrapper function for handling fastpredsolvinput and output 
    df = df.copy()
    df.rename(columns={solute_smiles_col:'solute_smiles'}, inplace=True)
    df['solvent_smiles']= solvent_smiles
    df['temperature']   = temp
    predval = fastsolv(df[['solute_smiles', 'solvent_smiles', 'temperature']])
    predval = predval.reset_index()
    predval.set_index(df.index, inplace=True)
    predval['pred_solubility [M]'] = 10**predval['predicted_logS']
    predval['uncertainty of pred_solubility [M]'] = 10**predval['predicted_logS_stdev']
    return predval

def sc6RunFastSolv(fd, debug):
    outfd    = MakeFolderWithCurrentFuncName(f'{fd}/results', allow_override=True)
    dmso     = 'CS(=O)C' # DMSO
    temp     = 298
    monomers = pd.read_csv(f'{fd}/data/yasuharasensei_monomers.tsv', sep='\t', index_col=0)
    predsolb = RunPredSolv(monomers, 'SMILES', dmso, temp)
    monomers = monomers.join(predsolb)
    monomers.to_csv(f'{outfd}/seled_monomers_fastsolv_pred.tsv', sep='\t')

    # ACD mols 
    acd = pd.read_csv(f'{fd}/results/sc5SelectMethacrylateACD/acd_methacrylates_1104.tsv', sep='\t')
    if debug:
        acd = acd.head(10)
    acd['solvent name'] = 'DMSO'
    predsolb = RunPredSolv(acd, 'SMILES', dmso, temp)
    acd       = acd.join(predsolb)
    acd.drop(columns='ROMol', inplace=True)
    outfname = f'{outfd}/acd_{len(acd)}_solubility_prediction'
    acd.to_csv(outfname + '.tsv', sep='\t')
    WriteDataFrameSmilesToXls(acd, 
                              smiles_colnames=['SMILES', 'neutral_SMILES_biggest'], 
                              retain_smiles_col=True, 
                              out_filename=outfname + '.xlsx',
                              cell_width=300, 
                              cell_height=300)

    # purchasable cpds
    regdb = pd.read_csv(f'{fd}/results/sc2ExtractMethaCrylate/db1_methacrylates_138.tsv', sep='\t', index_col=0)
    if debug:
        regdb = regdb.head(10)
    regdb['solvent_name']='DMSO'
    predsolb    = RunPredSolv(regdb, 'neutral_SMILES_biggest', dmso, temp)
    regdb       = regdb.join(predsolb)
    regdb.drop(columns='ROMol', inplace=True)

    outfname = f'{outfd}/regeantdb_{len(regdb)}_solubility_prediction'
    acd.to_csv(outfname + '.tsv', sep='\t')
    WriteDataFrameSmilesToXls(regdb, 
                              smiles_colnames=['SMILES', 'neutral_SMILES_biggest'], 
                              retain_smiles_col=True, 
                              out_filename=outfname + '.xlsx',
                              cell_width=300, 
                              cell_height=300)    

    
if __name__ =='__main__':
    bf = Path(__file__).parents[1]
    sc6RunFastSolv(bf, debug=False)
