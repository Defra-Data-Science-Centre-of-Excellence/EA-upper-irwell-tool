import sys
import itertools

import numpy as np
import pandas as pd


def calculate_pv_wlb(
    household_risk_changes,
    rofrs_damages_path,
    duration_of_benefits_DoB_period=50,
    annual_damages_avoided_compared_with_low_risk=np.array([0, 59, 294, 1000, 1589]),
    discount_factors_path = "/dbfs/mnt/lab/unrestricted/irwell_data/longterm_standard_discount_factor.csv",
    rofrs_damages_sheet_name='RoFRS Damages',
):
    """
    Calculate pv whole life benefits (WLB).

    This function follows the approach suggested by the EA using the "RoFRS Damages"
    worksheet in the per-community (Excel) workbooks.

    The calculation accounts for duration of benefits and discount factor by following
    the logic in the PF calculator.
    
    Notes:
        - There is an assumption that a risk category does not become more severe, i.e.
          there is always a positive effect after the duration of benefits. This may
          need to be reconsidered if climate change is incorporated explicitly.
    
    Args:
        ...
    
    Returns:
        float: pv whole life benefits
    
    TODO:
        - Still need to check with EA whether this is correct.
    
    """
    discount_factors = pd.read_csv(discount_factors_path)
    rofrs_damages = _get_rofrs_damages(rofrs_damages_path, rofrs_damages_sheet_name)
    
    risk_categories = [
        'low_risk', 'moderate_risk', 'intermediate_risk', 'significant_risk', 
        'very_significant_risk', 
    ]
    
    pv_wlb = 0.0
    for initial_risk, new_risk in itertools.combinations(reversed(risk_categories), r=2):
        i = risk_categories.index(initial_risk) + 1
        j = risk_categories.index(new_risk) + 1
        
        if (initial_risk, new_risk) in household_risk_changes.keys():
            if household_risk_changes[(initial_risk, new_risk)] > 0:
                pv_wlb += np.sum(rofrs_damages[j:i])
    
    dob_cumul_discount_factor = _get_cumul_discount_factor(
        discount_factors, duration_of_benefits_DoB_period
    )
    pv_wlb *= dob_cumul_discount_factor
    
    return pv_wlb


def _get_rofrs_damages(path, sheet_name):
    df1 = pd.read_excel(path, sheet_name=sheet_name, usecols='D:H', skiprows=1, nrows=4)
    df1['ID'] = range(len(df1))
    df2 = pd.read_excel(path, sheet_name=sheet_name, usecols='D:H', skiprows=7, nrows=4)
    df2['ID'] = range(3, 3 + len(df2))
    df3 = pd.read_excel(path, sheet_name=sheet_name, usecols='D:H', skiprows=13, nrows=4)
    df3['ID'] = range(6,9)
    df = pd.concat([df1, df2, df3], axis=0)
    df = df.iloc[:, :-1].sum(axis=0).to_numpy()
    return df


def calculate_gia(
    # -------------------------------------------------------------------------
    # Mandatory inputs
    
    pv_costs_for_approval,  #  Section 3 (E31): storage (m3) * cost (£/m3)
    pv_WLB_for_appraisal_period,  # Section 4 (E37): derived via "RoFRS Damages"
    # TODO: Check how pv_WLB_for_appraisal_period is calculated via "RoFRS Damages"
    num_households_at_risk_today_5a,  # Section 5A (F46:J48)
    num_households_at_risk_after_duration_of_benefits_5a,  # Section 5A (F51:J53)
    # TODO: Check if 5b tables should be mandatory inputs or if they are derived
    
    # -------------------------------------------------------------------------
    # Optional inputs

    # Section 1
    project_name=None,
    national_project_number=None,
    date_of_PF_calculator=None,
    lead_RMA=None,
    FCERM_GIA_applicant_type='Environment Agency',
    project_stage='SOC',
    option_reference=None,
    
    # Section 2
    confirmed_strategic_approach='Yes',
    
    # Section 3
    pv_future_costs=None,
    
    # Section 4
    duration_of_benefits_DoB_period=50,
    pv_WLB_DoB_OM1A=None,
    people_related_impacts_due_to_measures_proposed_DoB_OM1B=None,
    
    # Section 5A
    annual_damages_avoided_compared_with_low_risk=np.array([0, 59, 294, 1000, 1589]),
    discount_factors_path = "/dbfs/mnt/lab/unrestricted/irwell_data/longterm_standard_discount_factor.csv",
    
    # Section 5B
    year_ready_for_service=2028,  # TODO: Check default
    # TODO: Check if 5b tables should be mandatory inputs or if they are derived
    num_households_at_risk_2040_5b=None,
    num_households_at_risk_after_duration_of_benefits_5b=None,
    
):
    """
    Calculate PV maximum eligible FCERM GIA via Defra PF Calculator.
    
    Denoting input arguments as mandatory and optional - all arguments are required, 
    but the mandatory ones definitely need to be set explicitly. The optional ones 
    could be left at defaults potentially (but should be checked).
    
    Expected dataframe formats for both num_households_at_risk_today_5a and 
    num_households_at_risk_after_duration_of_benefits_5a are:
    
    ```
    num_households_at_risk_today_5a = pd.DataFrame({
        'deprivation_level': ['20% most deprived', '21% to 40% most deprived', '60% least deprived'],
        'low_risk': [0, 0, 0],  # keep fixed (protected in worksheet)
        'moderate_risk': [0, 0, 0],  # three entries, one per deprivation level
        'intermediate_risk': [0, 0, 0],  # as above ...
        'significant_risk': [0, 0, 0],
        'very_significant_risk': [0, 0, 0],
    })
    ```
    
    Args:
        ...
    
    Notes:
        - In Section 3 inputs, we are taking pv_costs_for_approval (E31) as an input. 
          This means that we do not need to take pv_appraisal_costs (E28), 
          pv_design_and_construction_costs (E29) or pv_risk_contingency (E30) as inputs.
        - pv_costs_for_approval should be calculated based on multiplying storage by a
          cost per storage (initially £20/cubic metre).
        - By default, pv_future_costs (still Section 3) will be calculated as 2.5% of
          the capital cost (pv_costs_for_approval).
        - In Section 4 inputs, pv_WLB_for_appraisal_period needs to be calculated
          externally prior to input using 'RoFRS Damages' sheet.
        - If pv_WLB_DoB_OM1A is None (default) then it will adopt the same value as 
          pv_WLB_appraisal_period.
        - For Section 5, discount factors should be read from file, as the table is
          relatively large. By default it is assumed that 
          'longterm_standard_discount_factor.csv' is in the same folder as this module.
        - Sections 6 and 7 are not currently implemented (though they were drafted in
          the notebook).
    
    Returns:
        tuple: project_benefit_cost_ratio (R8), pv_max_eligible_fcerm_gia (E23)
    
    """
    # ---
    # Assign defaults
    
    # Section 3
    if pv_future_costs is None:
        # TODO: Check that using correct variable here
        pv_future_costs = pv_costs_for_approval * 0.025
    
    if pv_WLB_DoB_OM1A is None:
        pv_WLB_DoB_OM1A = pv_WLB_for_appraisal_period
    
    # Section 4
    if people_related_impacts_due_to_measures_proposed_DoB_OM1B is None:
        people_related_impacts_due_to_measures_proposed_DoB_OM1B = pv_WLB_for_appraisal_period * 0.2
    
    # Section 5B
    # TODO: Clarify how these inputs are supposed to be approximated (25%?)
    if num_households_at_risk_2040_5b is None:
        raise NotImplementedError
    
    if num_households_at_risk_after_duration_of_benefits_5b is None:
        raise NotImplementedError
    
    # ---
    # Read lookup tables
    
    discount_factors = pd.read_csv(discount_factors_path)
    
    # ---
    # Calculations
    
    # Sections 3 & 1 - benefit-cost ratio
    pv_WLC_over_duration_of_benefits = pv_costs_for_approval + pv_future_costs  # E33: costs
    project_benefit_cost_ratio = pv_WLB_DoB_OM1A / pv_WLC_over_duration_of_benefits  # R8
    
    # Section 5A
    pv_qual_benefits_20pc_most_deprived_5a, \
        pv_qual_benefits_21to40pc_most_deprived_5a, \
        pv_qual_benefits_60pc_least_deprived_5a = _calculate_5a(
            num_households_at_risk_today_5a, 
            num_households_at_risk_after_duration_of_benefits_5a,
            discount_factors,
            duration_of_benefits_DoB_period,
            annual_damages_avoided_compared_with_low_risk,
    )
    
    # Section 5B
    pv_qual_benefits_20pc_most_deprived_5b, \
        pv_qual_benefits_21to40pc_most_deprived_5b, \
        pv_qual_benefits_60pc_least_deprived_5b = _calculate_5b(
            year_ready_for_service,
            num_households_at_risk_2040_5b, 
            num_households_at_risk_after_duration_of_benefits_5b,
            discount_factors,
            duration_of_benefits_DoB_period,
            annual_damages_avoided_compared_with_low_risk,
        )
    
    # Section 8
    total_qualifying_benefits, total_eligible_fcerm_gia = _calculate_8(
        people_related_impacts_due_to_measures_proposed_DoB_OM1B,
        pv_qual_benefits_20pc_most_deprived_5a,
        pv_qual_benefits_21to40pc_most_deprived_5a,
        pv_qual_benefits_60pc_least_deprived_5a,
        pv_qual_benefits_20pc_most_deprived_5b,
        pv_qual_benefits_21to40pc_most_deprived_5b,
        pv_qual_benefits_60pc_least_deprived_5b,
        pv_WLB_DoB_OM1A,  # input that is set as pv_WLB_for_appraisal_period by default
    )
    
    # Section 2 - PV max eligible FCERM GIA (E23)
    pv_max_eligible_fcerm_gia = 0
    if year_ready_for_service + duration_of_benefits_DoB_period < 2040:
        print(
            f'OM2 (2040) FCERM GIA eligibility is not applicable for:',
            f'year_ready_for_service = {year_ready_for_service} and DOB period =', 
            f'{duration_of_benefits_DoB_period} years'
        )
    else:
        if project_benefit_cost_ratio >= 1:
            if project_stage in ['Pre-SOC', 'SOC', 'Change (before OBC)']:
                pv_max_eligible_fcerm_gia = total_eligible_fcerm_gia
            else:
                raise ValueError(f'Economic summary required for {project_stage}')
        else:
            print(
                f'Low benefit-cost ratio ({project_benefit_cost_ratio}): if higher then PV'
                f'max eligible FCERM GIA could be: {total_eligible_fcerm_gia}'
            )
    
    return project_benefit_cost_ratio, pv_max_eligible_fcerm_gia


def _get_cumul_discount_factor(discount_factors, year):
    # TODO: Check that merge_asof works correctly as a lookup function here
    df = pd.merge_asof(
        pd.DataFrame({'Year': [year]}), discount_factors, on='Year', direction='nearest',
    )
    cumul_factor = df['Cumulative'].to_numpy()[0]
    return cumul_factor


def _pv_qual_benefits_5a(change_n_risk, damages_avoided, dob_cumul_discount_factor):
    pv_qual_benefits = -(
        np.sum(change_n_risk * damages_avoided) * dob_cumul_discount_factor
    )
    return pv_qual_benefits
    
    
def _calculate_5a(
    num_households_at_risk_today_5a, 
    num_households_at_risk_after_duration_of_benefits_5a,
    discount_factors,
    duration_of_benefits_DoB_period,
    annual_damages_avoided_compared_with_low_risk,
):
    n_risk_today = num_households_at_risk_today_5a
    n_risk_after_dob = num_households_at_risk_after_duration_of_benefits_5a
    damages_avoided = annual_damages_avoided_compared_with_low_risk
    
    if not np.all(n_risk_today.iloc[:, 1:].sum(axis=1) == n_risk_after_dob.iloc[:, 1:].sum(axis=1)):
        raise ValueError('Total households at risk today and after project are not equal.')
    
    change_n_risk = n_risk_after_dob.copy()
    change_n_risk.iloc[:, 1:] = 0
    change_n_risk.iloc[:, 1:] = n_risk_after_dob.iloc[:, 1:] - n_risk_today.iloc[:, 1:]

    dob_cumul_discount_factor = _get_cumul_discount_factor(
        discount_factors, duration_of_benefits_DoB_period,
    )
    
    pv_qual_benefits_20pc_most_deprived = _pv_qual_benefits_5a(
        change_n_risk.iloc[0, 1:], damages_avoided, dob_cumul_discount_factor,
    )
    pv_qual_benefits_21to40pc_most_deprived = _pv_qual_benefits_5a(
        change_n_risk.iloc[1, 1:], damages_avoided, dob_cumul_discount_factor,
    )
    pv_qual_benefits_60pc_least_deprived = _pv_qual_benefits_5a(
        change_n_risk.iloc[2, 1:], damages_avoided, dob_cumul_discount_factor,
    )
    
    return pv_qual_benefits_20pc_most_deprived, pv_qual_benefits_21to40pc_most_deprived, \
        pv_qual_benefits_60pc_least_deprived


def _pv_qual_benefits_5b(
    year_ready_for_service, duration_of_benefits_DoB_period, change_n_risk, 
    damages_avoided, cumul_discount_factor_1, cumul_discount_factor_2,
):
    if (year_ready_for_service + duration_of_benefits_DoB_period) < 2041:
        pv_qual_benefits = 0
    else:
        term_1 = -(
            np.sum(change_n_risk * damages_avoided) * cumul_discount_factor_1
        )
        term_2 = -(
            np.sum(change_n_risk * damages_avoided) * cumul_discount_factor_2
        )
        pv_qual_benefits = term_1 - term_2
    
    return pv_qual_benefits

    
def _calculate_5b(
    year_ready_for_service,
    num_households_at_risk_2040_5b, 
    num_households_at_risk_after_duration_of_benefits_5b,
    discount_factors,
    duration_of_benefits_DoB_period,
    annual_damages_avoided_compared_with_low_risk,
):
    n_risk_2040 = num_households_at_risk_2040_5b
    n_risk_after_dob = num_households_at_risk_after_duration_of_benefits_5b
    damages_avoided = annual_damages_avoided_compared_with_low_risk
    
    if not np.all(n_risk_2040.iloc[:, 1:].sum(axis=1) == n_risk_after_dob.iloc[:, 1:].sum(axis=1)):
        raise ValueError('Total households at risk today and after project are not equal.')
    
    change_n_risk = n_risk_after_dob.copy()
    change_n_risk.iloc[:, 1:] = 0
    change_n_risk.iloc[:, 1:] = n_risk_after_dob.iloc[:, 1:] - n_risk_2040.iloc[:, 1:]

    cumul_discount_factor_1 = _get_cumul_discount_factor(
        discount_factors, duration_of_benefits_DoB_period,
    )
    cumul_discount_factor_2 = _get_cumul_discount_factor(
        discount_factors, 2040 - year_ready_for_service,
    )
    
    pv_qual_benefits_20pc_most_deprived = _pv_qual_benefits_5b(
        year_ready_for_service, duration_of_benefits_DoB_period, 
        change_n_risk.iloc[0, 1:], damages_avoided, 
        cumul_discount_factor_1, cumul_discount_factor_2,
    ) 
    pv_qual_benefits_21to40pc_most_deprived = _pv_qual_benefits_5b(
        year_ready_for_service, duration_of_benefits_DoB_period, 
        change_n_risk.iloc[1, 1:], damages_avoided, 
        cumul_discount_factor_1, cumul_discount_factor_2,
    ) 
    pv_qual_benefits_60pc_least_deprived = _pv_qual_benefits_5b(
        year_ready_for_service, duration_of_benefits_DoB_period, 
        change_n_risk.iloc[2, 1:], damages_avoided, 
        cumul_discount_factor_1, cumul_discount_factor_2,
    )
    
    return pv_qual_benefits_20pc_most_deprived, pv_qual_benefits_21to40pc_most_deprived, \
        pv_qual_benefits_60pc_least_deprived


def _calculate_8(
    people_related_impacts_due_to_measures_proposed_DoB_OM1B,
    pv_qual_benefits_20pc_most_deprived_5a,
    pv_qual_benefits_21to40pc_most_deprived_5a,
    pv_qual_benefits_60pc_least_deprived_5a,
    pv_qual_benefits_20pc_most_deprived_5b,
    pv_qual_benefits_21to40pc_most_deprived_5b,
    pv_qual_benefits_60pc_least_deprived_5b,
    pv_WLB_DoB_OM1A,  # input that is set as pv_WLB_for_appraisal_period by default
):
    main_table = pd.DataFrame({
        'om': ['om1a', 'om1b', 'om2', 'om2', 'om2', 'om3', 'om3', 'om3', 'om4', 'om4'],
        'deprivation': [
            'overall', 'people related', '20% most', '21% to 40%', '60% least', '20% most',
            '21% to 40%', '60% least', 'habitat', 'rivers',
        ],
        'qualifying_benefits': np.zeros(10),
        'payment_rate': [6, 20, 45, 30, 20, 45, 30, 20, 20, 20],
        'eligible_fcerm_gia': np.zeros(10),
    })

    # OM1b
    main_table.loc[
        (main_table['om'] == 'om1b') & (main_table['deprivation'] == 'people related'),
        'qualifying_benefits'
    ] = people_related_impacts_due_to_measures_proposed_DoB_OM1B

    # OM2
    main_table.loc[
        (main_table['om'] == 'om2') & (main_table['deprivation'] == '20% most'),
        'qualifying_benefits'
    ] = pv_qual_benefits_20pc_most_deprived_5a + pv_qual_benefits_20pc_most_deprived_5b
    main_table.loc[
        (main_table['om'] == 'om2') & (main_table['deprivation'] == '21% to 40%'),
        'qualifying_benefits'
    ] = pv_qual_benefits_21to40pc_most_deprived_5a + pv_qual_benefits_21to40pc_most_deprived_5b
    main_table.loc[
        (main_table['om'] == 'om2') & (main_table['deprivation'] == '60% least'),
        'qualifying_benefits'
    ] = pv_qual_benefits_60pc_least_deprived_5a + pv_qual_benefits_60pc_least_deprived_5b
    
    # ---
    # For testing against Hindley example only
    # main_table.loc[
    #     (main_table['om'] == 'om4') & (main_table['deprivation'] == 'habitat'),
    #     'qualifying_benefits'
    # ] = 326705
    # ---

    # OM1a
    if pv_WLB_DoB_OM1A == 0:
        val = 0
    else:
        val = max(pv_WLB_DoB_OM1A - main_table['qualifying_benefits'].iloc[1:].sum(), 0)
        # in worksheet zero would be 'Ltd by high OM1b,2,3,4 values'
    main_table.loc[
        (main_table['om'] == 'om1a') & (main_table['deprivation'] == 'overall'), 'qualifying_benefits'
    ] = val

    # Eligible FCERM GIA
    main_table['eligible_fcerm_gia'] = np.maximum(
        main_table['qualifying_benefits'] * (main_table['payment_rate'] / 100),
        0
    )

    # Total numbers
    total_qualifying_benefits = main_table['qualifying_benefits'].sum()
    total_eligible_fcerm_gia = main_table['eligible_fcerm_gia'].sum()
    
    return total_qualifying_benefits, total_eligible_fcerm_gia


def calculate_pv_wlb__interim(
    num_households_at_risk_today_5a, 
    num_households_at_risk_after_duration_of_benefits_5a,
    duration_of_benefits_DoB_period=50,
    annual_damages_avoided_compared_with_low_risk=np.array([0, 59, 294, 1000, 1589]),
    discount_factors_path = "/dbfs/mnt/lab/unrestricted/irwell_data/longterm_standard_discount_factor.csv",
):
    """
    Calculate pv whole life benefits (WLB).

    Note that this is likely an interim function, as it does not follow the EA-suggested
    approach (not yet clear on this). The calculation implemented uses the sum of cells
    D97:D99 in the "PF calculator" to approximate the WLB using the calculations 
    undertaken in Section 5A. We assume for now that Section 5B is not relevant (i.e. 
    it contributes zero to the pv WLB calculations in cells D97:D99). 
    
    Section 5 calculations may provide (for now) a clearer means of associating the 
    changes in risk categories with annual damages avoided. However, this needs to be
    checked and reconciled with "RoFRS Damages" worksheets available per community.

    The calculation accounts for duration of benefits and discount factor by following
    the logic in the PF calculator.
    
    TODO:
        - Check whether this approach should be retained.
        - Check if/how Section 5B should fit in.
    
    """
    discount_factors = pd.read_csv(discount_factors_path)
    
    pv_qual_benefits_20pc_most_deprived_5a, \
        pv_qual_benefits_21to40pc_most_deprived_5a, \
        pv_qual_benefits_60pc_least_deprived_5a = _calculate_5a(
            num_households_at_risk_today_5a, 
            num_households_at_risk_after_duration_of_benefits_5a,
            discount_factors,
            duration_of_benefits_DoB_period,
            annual_damages_avoided_compared_with_low_risk,
        )
    
    pv_wlb = np.sum([
        pv_qual_benefits_20pc_most_deprived_5a,
        pv_qual_benefits_21to40pc_most_deprived_5a,
        pv_qual_benefits_60pc_least_deprived_5a,
    ])
    
    return pv_wlb
